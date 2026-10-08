#include "models.h"

#include <algorithm>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif
#include "llama-impl.h"

#include "ggml-backend.h"

#include <cstdlib>
#include "llama-memory-hybrid-idx.h"
#include "llama-memory-recurrent.h"

#include <algorithm>
#include <cinttypes>

// [TAG_QWEN4_REIMPLEMENT]
// TODO: this graph implementation is pending complete reimplementation - do not use it as a reference

// bad metadata must be catchable: GGML_ASSERT aborts the whole process
static void qwen4exp_require_nonzero(const llama_model_loader & ml, llm_kv kid, uint32_t value) {
    if (value == 0) {
        throw std::runtime_error(format("%s must be greater than zero, got %u", ml.llm_kv(kid).c_str(), value));
    }
}

// get_arr() copies a short array as-is, leaving a zero tail the n-gram hash silently drops
static void qwen4exp_require_arr_len(llama_model_loader & ml, llm_kv kid, uint32_t n_min) {
    uint32_t n_arr = 0;
    ml.get_arr_n(kid, n_arr, true);
    if (n_arr < n_min) {
        throw std::runtime_error(format("%s has %u entries, but at least %u are required",
                                        ml.llm_kv(kid).c_str(), n_arr, n_min));
    }
}

void llama_model_qwen4exp::load_arch_hparams(llama_model_loader & ml) {
    // NextN/MTP: an extra decoder block appended past the trunk. Read this first, since
    // n_layer() == n_layer_all - n_layer_nextn feeds every per-layer array below.
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all && "n_layer_nextn must be < block_count");

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,       hparams.f_norm_rms_eps);

    ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS,    hparams.rope_sections, 4, true);

    ml.get_key(LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    ml.get_key(LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_CONV_KERNEL,    hparams.ssm_d_conv);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_INNER_SIZE,     hparams.ssm_d_inner);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_STATE_SIZE,     hparams.ssm_d_state);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_TIME_STEP_RANK, hparams.ssm_dt_rank);
    qwen4exp_require_nonzero(ml, LLM_KV_SSM_GROUP_COUNT,    hparams.ssm_n_group);

    // HC; low_rank is qwen4exp-specific, DeepSeek-V4 leaves it absent (full rank)
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,    hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    // a count of 1 has nothing to mix: transformers configuration_qwen4_exp.py:196, vLLM
    // config.py:49 and SGLang configs/qwen4_exp.py:38 all raise on hc_count <= 1
    if (hparams.dsv4_hc_mult <= 1) {
        throw std::runtime_error(format("%s must be greater than one, got %u",
                                        ml.llm_kv(LLM_KV_HYPER_CONNECTION_COUNT).c_str(), hparams.dsv4_hc_mult));
    }
    qwen4exp_require_nonzero(ml, LLM_KV_HYPER_CONNECTION_LOW_RANK, hparams.hc_low_rank);
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    qwen4exp_require_nonzero(ml, LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key_or_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios, hparams.n_layer_all, false);

    // halo-hybrid: the MTP block carries its own indexer, but the converter writes ratio 0 for it (dense, see
    // graph_mtp). LLAMA_MTP_QSA=1 gives it the trunk's ratio so the draft attends through QSA (opt-in: measured worse).
    {
        static const bool mtp_qsa = getenv("LLAMA_MTP_QSA") != nullptr && atoi(getenv("LLAMA_MTP_QSA")) != 0;
        if (mtp_qsa && hparams.n_layer_nextn > 0) {
            uint32_t r_trunk = 0;
            for (uint32_t il = 0; il < hparams.n_layer() && r_trunk == 0; ++il) { r_trunk = hparams.dsv4_compress_ratios[il]; }
            for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
                if (hparams.dsv4_compress_ratios[il] == 0) { hparams.dsv4_compress_ratios[il] = r_trunk; }
            }
        }
    }

    // PLE n-gram hash embeddings; if the key group is absent every field stays zero
    hparams.is_ple_impl.reset();
    hparams.ple_n_heads = 0;

    uint32_t n_ple = 0;
    ml.get_arr_n(LLM_KV_PLE_LAYERS, n_ple, false);
    if (n_ple > 0) {
        std::vector<uint32_t> ple_layers;
        ml.get_arr(LLM_KV_PLE_LAYERS, ple_layers);
        if (n_ple != 1) {
            // hparams holds one set of hash constants, so several PLE modules cannot be represented
            throw std::runtime_error(format("%s lists %u layers, but only one PLE layer is supported",
                                            ml.llm_kv(LLM_KV_PLE_LAYERS).c_str(), n_ple));
        }
        for (uint32_t il : ple_layers) {
            if (il >= hparams.n_layer_all) {
                throw std::runtime_error(format("PLE layer %u is out of range", il));
            }
            hparams.is_ple_impl.set(il);
        }

        ml.get_key(LLM_KV_PLE_NGRAM_SIZE,      hparams.ple_ngram_size);
        ml.get_key(LLM_KV_PLE_HEADS_PER_NGRAM, hparams.ple_heads_per_ngram);
        ml.get_key(LLM_KV_PLE_CONV_KERNEL,     hparams.ple_conv_kernel);
        ml.get_key(LLM_KV_PLE_EOS_TOKEN_ID,    hparams.ple_eos_token_id);
        // optional: files written before this key fall back to the EOS token
        ml.get_key(LLM_KV_PLE_IMAGE_TOKEN_ID,  hparams.ple_image_token_id, false);
        ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER, hparams.n_embd_per_layer);
        qwen4exp_require_nonzero(ml, LLM_KV_PLE_CONV_KERNEL,             hparams.ple_conv_kernel);
        qwen4exp_require_nonzero(ml, LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  hparams.n_embd_per_layer);

        hparams.ple_n_heads  = (hparams.ple_ngram_size - 1) * hparams.ple_heads_per_ngram;
        hparams.ple_head_dim = hparams.n_embd_per_layer;
        if (hparams.ple_ngram_size < 2 || hparams.ple_ngram_size > LLAMA_MAX_PLE_NGRAM) {
            throw std::runtime_error(format("PLE n-gram size %u is out of range", hparams.ple_ngram_size));
        }
        if (hparams.ple_n_heads == 0 || hparams.ple_n_heads > LLAMA_MAX_PLE_HEADS) {
            throw std::runtime_error(format("PLE head count %u is out of range", hparams.ple_n_heads));
        }

        qwen4exp_require_arr_len(ml, LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_ngram_size);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_OFFSETS,      hparams.ple_n_heads);
        qwen4exp_require_arr_len(ml, LLM_KV_PLE_HEAD_VOCAB_SIZES,  hparams.ple_n_heads);

        ml.get_arr(LLM_KV_PLE_LAYER_MULTIPLIERS, hparams.ple_layer_multipliers);

        // the file stores the head ranges as uint64, so read at that width and narrow to the int32 the gather uses
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_offsets     = {};
        std::array<uint64_t, LLAMA_MAX_PLE_HEADS> head_vocab_sizes = {};
        ml.get_arr(LLM_KV_PLE_HEAD_OFFSETS,     head_offsets);
        ml.get_arr(LLM_KV_PLE_HEAD_VOCAB_SIZES, head_vocab_sizes);
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            if (head_vocab_sizes[h] == 0 ||
                head_offsets[h]     > INT32_MAX ||
                head_vocab_sizes[h] > INT32_MAX ||
                head_offsets[h] + head_vocab_sizes[h] > INT32_MAX) {
                throw std::runtime_error(format("PLE head %u range does not fit the int32 row index", h));
            }
            hparams.ple_head_offsets[h]     = (uint32_t) head_offsets[h];
            hparams.ple_head_vocab_sizes[h] = (uint32_t) head_vocab_sizes[h];
        }
    }

    // linear attention everywhere except every full_attention_interval-th layer
    if (!ml.get_key_or_arr(LLM_KV_ATTENTION_RECURRENT_LAYERS, hparams.is_recr_impl, hparams.n_layer_all, false)) {
        uint32_t full_attn_interval = 4;
        ml.get_key(LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval, false);
        qwen4exp_require_nonzero(ml, LLM_KV_FULL_ATTENTION_INTERVAL, full_attn_interval);
        for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
            hparams.is_recr_impl[i] = (i < hparams.n_layer()) && ((i + 1) % full_attn_interval != 0);
        }
    }

    // the PLE conv history is a row of the recurrent cache, which linear layers alone have
    for (uint32_t i = 0; i < hparams.n_layer_all; ++i) {
        if (hparams.is_ple(i) && !hparams.is_recr(i)) {
            throw std::runtime_error(format("PLE layer %u is not a linear attention layer", i));
        }
    }

    switch (hparams.n_layer()) {
        case 48: type = LLM_TYPE_A3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_qwen4exp::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t hc_lr  = hparams.hc_low_rank;

    // a draft-only export declares the full block count but ships the MTP block alone,
    // so the trunk is described and absent. same probe as qwen35.
    const bool mtp_only    = (hparams.n_layer_nextn > 0) && (ml.get_weight("blk.0.hc_attn_norm.weight") == nullptr);
    const int  trunk_flags = mtp_only ? TENSOR_NOT_REQUIRED : 0;
    hparams.mtp_only = mtp_only;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, 0);

    // there is no output_norm: the final hyper-connection mixer carries it. the MTP head
    // has its own in nextn.hc_head_*, so a draft-only file does not carry these
    hc_head_norm = create_tensor(tn(LLM_TENSOR_HC_HEAD_NORM, "weight"), { hc_dim }, trunk_flags);
    hc_head_down = create_tensor(tn(LLM_TENSOR_HC_HEAD_DOWN, "weight"), { hc_dim, hc_lr }, trunk_flags);
    hc_head_up   = create_tensor(tn(LLM_TENSOR_HC_HEAD_UP,   "weight"), { hc_lr, hc_dim }, trunk_flags);

    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), { n_embd, n_vocab }, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), { n_embd, n_vocab }, TENSOR_DUPLICATED);
    }

    // flat [ple_head_dim, n_rows] gather target
    if (hparams.ple_n_heads > 0) {
        // the head ranges are what the gather indexes, so they set the minimum row count
        int64_t ple_rows = 0;
        for (uint32_t h = 0; h < hparams.ple_n_heads; ++h) {
            ple_rows = std::max(ple_rows, (int64_t) hparams.ple_head_offsets[h] + hparams.ple_head_vocab_sizes[h]);
        }

        // the converter pads the table; a model synthesised from metadata has no tensor to ask
        const std::string ple_name = tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight").str();
        if (const auto * ple_w = ml.get_weight(ple_name.c_str())) {
            if (ple_w->tensor->ne[1] < ple_rows) {
                throw std::runtime_error(format("%s has %" PRId64 " rows, too few for the PLE head ranges (%" PRId64 ")",
                                                ple_name.c_str(), ple_w->tensor->ne[1], ple_rows));
            }
            ple_rows = ple_w->tensor->ne[1];
        }

        per_layer_tok_embd = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"),
                                           { hparams.ple_head_dim, ple_rows }, TENSOR_READ_LAZY);
    }

    // MTP tensors sit in the trailing blocks; skip them entirely unless a draft head was asked for
    const int mtp_flags = !ml.load_mtp ? TENSOR_SKIP : 0;

    for (int il = 0; il < (int) hparams.n_layer_all; ++il) {
        auto & layer = layers[il];

        // the MTP block is structurally a trunk block: is_recr()/is_ple() are both false past
        // the trunk, so it takes the full-attention + MoE path below with no special casing
        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        const int64_t n_ff_exp   = hparams.n_ff_exp(il) ? hparams.n_ff_exp(il) : n_ff / n_expert_used;
        const int64_t n_ff_shexp = hparams.n_ff_shexp ? hparams.n_ff_shexp : n_ff;

        const int64_t head_k_dim = hparams.ssm_d_state;
        const int64_t head_v_dim = hparams.ssm_d_state;
        const int64_t n_k_heads  = hparams.ssm_n_group;
        const int64_t n_v_heads  = hparams.ssm_dt_rank;
        const int64_t key_dim    = head_k_dim * n_k_heads;
        const int64_t value_dim  = head_v_dim * n_v_heads;
        const int64_t conv_dim   = key_dim * 2 + value_dim;

        // two HC modules per layer: before the token mixer, before the MoE
        layer.hc_attn_norm   = create_tensor(tn(LLM_TENSOR_HC_ATTN_NORM,   "weight", il), { hc_dim }, flags);
        layer.hc_attn_down   = create_tensor(tn(LLM_TENSOR_HC_ATTN_DOWN,   "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_attn_up     = create_tensor(tn(LLM_TENSOR_HC_ATTN_UP,     "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_attn_inject = create_tensor(tn(LLM_TENSOR_HC_ATTN_INJECT, "weight", il), { hc_dim, hc }, flags);
        layer.hc_ffn_norm    = create_tensor(tn(LLM_TENSOR_HC_FFN_NORM,    "weight", il), { hc_dim }, flags);
        layer.hc_ffn_down    = create_tensor(tn(LLM_TENSOR_HC_FFN_DOWN,    "weight", il), { hc_dim, hc_lr }, flags);
        layer.hc_ffn_up      = create_tensor(tn(LLM_TENSOR_HC_FFN_UP,      "weight", il), { hc_lr, hc_dim }, flags);
        layer.hc_ffn_inject  = create_tensor(tn(LLM_TENSOR_HC_FFN_INJECT,  "weight", il), { hc_dim, hc }, flags);

        if (!hparams.is_recr(il)) {
            // full attention: wq holds [q|gate] interleaved per head
            create_tensor_qkv(layer, il, n_embd, n_embd_head_k * n_head * 2, n_embd_k_gqa, n_embd_v_gqa, flags);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), { n_embd_head_k * n_head, n_embd }, flags);

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", il), { n_embd_head_k }, flags);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", il), { n_embd_head_k }, flags);

            const int64_t idx_dim = hparams.indexer_head_size;
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", il), { n_embd, hparams.indexer_n_head * idx_dim }, flags);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", il), { n_embd, idx_dim }, flags);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", il), { idx_dim }, flags);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", il), { idx_dim }, flags);
        } else {
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", il), { n_embd, key_dim * 2 + value_dim }, flags);
            layer.wqkv_gate  = create_tensor(tn(LLM_TENSOR_ATTN_GATE,  "weight", il), { n_embd, value_dim }, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", il), { hparams.ssm_d_conv, conv_dim }, flags);
            layer.ssm_dt     = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_a      = create_tensor(tn(LLM_TENSOR_SSM_A_NOSCAN,         il), { hparams.ssm_dt_rank }, flags);
            layer.ssm_beta   = create_tensor(tn(LLM_TENSOR_SSM_BETA,   "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_alpha  = create_tensor(tn(LLM_TENSOR_SSM_ALPHA,  "weight", il), { n_embd, n_v_heads }, flags);
            layer.ssm_norm   = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", il), { head_v_dim }, flags);
            layer.ssm_out    = create_tensor(tn(LLM_TENSOR_SSM_OUT,    "weight", il), { value_dim, n_embd }, flags);
        }

        if (hparams.is_ple(il)) {
            layer.ple_key        = create_tensor(tn(LLM_TENSOR_PLE_KEY,        "weight", il), { n_embd, hc_dim }, flags);
            layer.ple_value      = create_tensor(tn(LLM_TENSOR_PLE_VALUE,      "weight", il), { n_embd, n_embd }, flags);
            layer.ple_norm_key   = create_tensor(tn(LLM_TENSOR_PLE_NORM_KEY,   "weight", il), { hc_dim }, flags);
            layer.ple_norm_query = create_tensor(tn(LLM_TENSOR_PLE_NORM_QUERY, "weight", il), { hc_dim }, flags);
            layer.ple_norm_conv  = create_tensor(tn(LLM_TENSOR_PLE_NORM_CONV,  "weight", il), { hc_dim }, flags);
            layer.ple_conv1d     = create_tensor(tn(LLM_TENSOR_PLE_CONV1D,     "weight", il), { hparams.ple_conv_kernel, hc_dim }, flags);
        }

        layer.ffn_gate_inp  = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,  "weight", il), { n_embd, n_expert }, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), { n_ff_exp, n_embd, n_expert }, flags);
        create_tensor_gate_up_exps(layer, il, n_embd, n_ff_exp, n_expert, flags);

        layer.ffn_gate_inp_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP_SHEXP, "weight", il), { n_embd }, flags);
        layer.ffn_gate_shexp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP,     "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_up_shexp       = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,       "weight", il), { n_embd, n_ff_shexp }, flags);
        layer.ffn_down_shexp     = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP,     "weight", il), { n_ff_shexp, n_embd }, flags);

        if (il < n_layer) {
            continue;
        }

        // NextN/MTP head. enorm/hnorm gate the two inputs; eh_proj is the checkpoint's
        // fc_embedding and fc_hidden fused side by side, so one matmul over
        // concat(e, h) computes fc_embedding@e + fc_hidden@h.
        layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), { n_embd }, flags);
        layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), { hc_dim }, flags);
        layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), { 2 * n_embd, n_embd }, flags);

        // the head's own output mixer, mirroring the trunk's hc_head_*: it collapses the
        // hc streams and stands in for the output norm, of which qwen4exp has none
        layer.nextn.hc_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_NORM, "weight", il), { hc_dim }, flags);
        layer.nextn.hc_head_down = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_DOWN, "weight", il), { hc_dim, hc_lr }, flags);
        layer.nextn.hc_head_up   = create_tensor(tn(LLM_TENSOR_NEXTN_HC_HEAD_UP,   "weight", il), { hc_lr, hc_dim }, flags);

        // qwen4exp sets mtp_use_dedicated_embeddings=false, so these are absent and the
        // head falls back to the trunk's embedding table and LM head
        layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), { n_embd, n_vocab }, flags | TENSOR_NOT_REQUIRED);
        // halo-hybrid: a reduced-vocabulary draft head. A draft file may carry its own nextn.shared_head_head over a
        // subset of the vocabulary plus `d2t` (I64, the target token id of each head row), as EAGLE-3 does: drafting
        // then reads ~n_draft rows instead of the full LM head, and verification still uses the full one.
        int64_t n_head_rows = n_vocab;
        if (const ggml_tensor * d2t_meta = ml.get_tensor_meta(tn(LLM_TENSOR_D2T).str().c_str())) {
            n_head_rows = d2t_meta->ne[0];
            d2t = create_tensor(tn(LLM_TENSOR_D2T), { n_head_rows }, 0);
            LLAMA_LOG_INFO("%s: MTP head over a reduced vocabulary: %lld of %lld tokens\n", __func__, (long long) n_head_rows, (long long) n_vocab);
        }
        layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), { n_embd, n_head_rows },
                flags | (n_head_rows == n_vocab ? TENSOR_NOT_REQUIRED : 0));
    }
}

std::unique_ptr<llm_graph_context> llama_model_qwen4exp::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// Hyper-connections keep hc parallel residual streams [n_embd, hc, T] in place of layer norms.
// Returns the mixed [n_embd, T] stream; `inject` gets the [hc, T] scatter weights.
ggml_tensor * llama_model_qwen4exp::graph::build_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    const int64_t nt     = x->ne[2];

    // grouped RMSNorm: rms_norm reduces over one residual stream, then the [hc_dim]
    // gamma scales all streams. Gammas were folded to (1 + w) by the converter.
    // The gamma is applied as a [n_embd, hc] broadcast over the 3-D tensor so that no reshape
    // sits between rms_norm and mul: the CUDA backend then fuses the two into one kernel.
    // (the weight view is expanded first so that RMS_NORM and MUL are consecutive graph nodes,
    // which is what the backend's rms_norm+mul fusion requires)
    ggml_tensor * w2 = ggml_reshape_2d(ctx0, w_norm, n_embd, hc);
    ggml_build_forward_expand(gf, w2);
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    xn = ggml_mul(ctx0, xn, w2);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo = build_lora_mm(w_down, xn);
    if (inject) {
        // built next to `lo` and pinned adjacent: both read xn, so the CUDA backend quantises it once
        *inject = build_lora_mm(w_inject, xn);
        cb(*inject, "hc_inject", il);
        ggml_build_forward_expand(gf, lo);
        ggml_build_forward_expand(gf, *inject);
    }
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f / (float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, build_lora_mm(w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // collapse the streams by their mean: sum the stream views directly (binary ops take a strided
    // src0), so there is no copy of stream 0; the three adds fuse into one kernel on CUDA
    ggml_tensor * mixed = nullptr;
    for (int64_t c = 0; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd) * hc,
                ggml_row_size(gated->type, n_embd) * c);
        mixed = mixed ? ggml_add(ctx0, mixed, s) : s;
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f / (float) hc);
    cb(mixed, "hc_mixed", il);

    return mixed;
}

ggml_tensor * llama_model_qwen4exp::graph::build_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f / (float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

llama_model_qwen4exp::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hparams.n_embd_head_v() == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.input_embed", -1);
    ggml_build_forward_expand(gf, inpL);

    auto * inp = build_inp_mem_hybrid();

    // qwen4exp always builds llama_memory_hybrid_idx, so this downcast is safe
    // the indexer cache inside it is absent when the GGUF has no indexer tensors
    const auto * mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp->mctx);

    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();
    if (mctx_idx) {
        GGML_ASSERT(mctx_idx->get_n_kv() == inp->mctx->get_attn()->get_n_kv() &&
                mctx_idx->get_kv_lo() == inp->mctx->get_attn()->get_kv_lo() &&
                "the indexer cache must track the attention cache cell for cell");
    }

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * ple_emb = nullptr;
    if (hparams.ple_n_heads > 0) {
        ple_emb = build_inp_ple(mctx_hyb);
        // make sure ple_emb and build_inp_embd are in the same graph split
        ggml_build_forward_expand(gf, ple_emb);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = res_hc;

        if (hparams.is_ple(il)) {
            res_hc = build_ple(inp->get_recr(), ple_emb, res_hc, il);
        }

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_hc_mix(res_hc,
                model.layers[il].hc_attn_norm,
                model.layers[il].hc_attn_down,
                model.layers[il].hc_attn_up,
                model.layers[il].hc_attn_inject,
                &inject, il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            cur = build_layer_attn(inp->get_attn(), mctx_hyb, cur, inp_pos, sections, il);
        }

        // an unmasked MTP export needs a hidden row for every token, so in that case the
        // gather is deferred until after t_h_nextn is taken below
        const bool gather_now = !cparams.embeddings_nextn || cparams.embeddings_nextn_masked;

        if (il == n_layer - 1 && inp_out_ids && gather_now) {
            // everything below is per token, so drop the rows that produce no output
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            inject = ggml_get_rows(ctx0, inject, inp_out_ids);

            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        cur = build_hc_mix(res_hc,
                model.layers[il].hc_ffn_norm,
                model.layers[il].hc_ffn_down,
                model.layers[il].hc_ffn_up,
                model.layers[il].hc_ffn_inject,
                &inject, il);

        cur = build_layer_ffn(cur, il);
        cb(cur, "ffn_out", il);

        res_hc = build_hc_combine(res_hc, cur, inject, il);

        // "l_last" is the layer output name that build_cvec and imatrix look for
        cb(res_hc, "l_last", il);
    }

    // The MTP head consumes the wide residual, before the head mixer collapses it. Export the
    // combine result itself rather than a reshape of it: a pure view gets no backend assignment
    // from the scheduler, and the readback in llama_context looks one up. It is contiguous, so
    // [n_embd, hc, rows] already has the [n_embd_out, rows] layout the reader expects, and it
    // carries exactly the right rows either way -- gathered above when masked, ungathered when not.
    if (cparams.embeddings_nextn) {
        cb(res_hc, "h_nextn", -1);
        res->t_h_nextn = res_hc;

        // deferred from the last layer: collapse to the output rows now that the export is taken
        if (!cparams.embeddings_nextn_masked && inp_out_ids) {
            res_hc = ggml_reshape_2d(ctx0, res_hc, n_embd*hc, res_hc->ne[2]);
            res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
            res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
        }
    }

    // the final mixer is the output norm: there is no separate one
    ggml_tensor * cur = build_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up,
            nullptr, nullptr, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

// LLM_GRAPH_TYPE_DECODER_MTP draft head for qwen4exp.
//
// The head folds the next token's embedding into the trunk's wide hyper-connection residual,
// runs one trunk-style block over it, and collapses the result with its own mixer before
// reusing the trunk's LM head. The wide post-block residual is exported as t_h_nextn so the
// speculative driver can feed it straight back in for the next draft step.
//
// The block attends densely by default. The trunk's QSA only prunes context past a 2048-token budget, so dense is a
// numerical superset; drafts are verified by the target either way. LLAMA_MTP_QSA=1 runs the block through the
// trunk's QSA path instead (its own indexer, hybrid memory with an indexer cache for this layer). Measured
// 2026-09-27 on Swift 1.5 Q8T, APU + R9700, 19-23K-token contexts: acceptance 0.759 dense vs 0.732 QSA, 43.6 vs 45.3
// ms/step, and the per-ubatch ingest is slower on the R9700 (dense FA is tuned there): so the head appears to have
// been trained dense and QSA stays opt-in.
llama_model_qwen4exp::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(model, params, no_build_t{}) {
    GGML_ASSERT(hparams.n_layer_nextn > 0 && "QWEN4EXP MTP requires n_layer_nextn > 0");
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "QWEN4EXP MTP currently only supports a single MTP block");
    GGML_ASSERT(ubatch.token && "QWEN4EXP MTP requires token input");

    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc * n_embd;
    GGML_ASSERT(hparams.n_embd_out() == (uint32_t) hc_dim && "QWEN4EXP MTP hidden width mismatch");

    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];

    GGML_ASSERT(layer.nextn.eh_proj     && "MTP block missing nextn.eh_proj");
    GGML_ASSERT(layer.nextn.enorm       && "MTP block missing nextn.enorm");
    GGML_ASSERT(layer.nextn.hnorm       && "MTP block missing nextn.hnorm");
    GGML_ASSERT(layer.nextn.hc_head_norm && "MTP block missing nextn.hc_head_norm");

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    auto inp = std::make_unique<llm_graph_input_embd_h>(hc_dim);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->embd);

    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd;
    ggml_tensor * tok_embd   = ggml_get_rows(ctx0, tok_embd_w, inp->tokens);
    cb(tok_embd, "mtp_tok_embd", il);

    ggml_tensor * h_state = ggml_reshape_3d(ctx0, inp->h, n_embd, hc, n_tokens);
    cb(h_state, "mtp_h_state", il);

    res->add_input(std::move(inp));

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // QSA draft (LLAMA_MTP_QSA): the context then carries the hybrid memory with an indexer cache for this layer
    const bool mtp_qsa = hparams.dsv4_compress_ratios[il] > 0;
    llm_graph_input_attn_kv * inp_attn = nullptr;
    const llama_memory_hybrid_idx_context * mctx_hyb = nullptr;
    if (mtp_qsa) {
        auto * inp_mem = build_inp_mem_hybrid();
        mctx_hyb = static_cast<const llama_memory_hybrid_idx_context *>(inp_mem->mctx);
        inp_attn = inp_mem->get_attn();
    } else {
        inp_attn = build_attn_inp_kv();
    }

    // grouped RMSNorm over the wide stream: normalise each hc stream, then scale the flattened
    // [hc_dim] vector with the head's gamma, exactly as build_hc_mix does
    ggml_tensor * h_norm = ggml_rms_norm(ctx0, h_state, hparams.f_norm_rms_eps);
    h_norm = ggml_reshape_2d(ctx0, h_norm, hc_dim, n_tokens);
    h_norm = ggml_mul(ctx0, h_norm, layer.nextn.hnorm);
    h_norm = ggml_reshape_3d(ctx0, h_norm, n_embd, hc, n_tokens);
    cb(h_norm, "mtp_hnorm", il);

    // the token embedding is shared across the streams, so broadcast it to hc copies
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    e_norm = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, e_norm, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(e_norm, "mtp_enorm", il);

    // eh_proj holds fc_embedding and fc_hidden side by side, so this one matmul is
    // fc_embedding @ e_norm + fc_hidden @ h_norm, applied to each stream independently.
    // Keeping the streams distinct here is the point of the hyper-connection residual:
    // pooling them before the projection would throw that away.
    ggml_tensor * concat = ggml_concat(ctx0, e_norm, h_norm, /*dim=*/ 0);
    cb(concat, "mtp_concat", il);

    // LLAMA_MTP_EH_PROJ_2D=1: one 2D GEMM over hc*n_tokens columns instead of n_tokens batched 4-column products.
    // Always-on it cost decode (acceptance 0.63 -> 0.62, -1%, 2026-09-27), but the prompt ingest's batched product
    // is n_tokens tiny GEMVs (51 ms per 4096-token ubatch on the R9700, 1.8 s of a 113K prefill): batches of at least
    // LLAMA_MTP_EH_PROJ_2D_MIN tokens (default 256; 0 = never) take the 2D GEMM, decode keeps the batched product
    static const bool eh_2d_all = getenv("LLAMA_MTP_EH_PROJ_2D") && atoi(getenv("LLAMA_MTP_EH_PROJ_2D")) > 0;
    static const int  eh_2d_min = getenv("LLAMA_MTP_EH_PROJ_2D_MIN") ? atoi(getenv("LLAMA_MTP_EH_PROJ_2D_MIN")) : 256;
    const bool eh_3d = !(eh_2d_all || (eh_2d_min > 0 && n_tokens >= eh_2d_min));
    ggml_tensor * res_hc;
    if (eh_3d) {
        res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
    } else {
        res_hc = build_lora_mm(layer.nextn.eh_proj, ggml_reshape_2d(ctx0, concat, concat->ne[0], hc*n_tokens), layer.nextn.eh_proj_s);
        res_hc = ggml_reshape_3d(ctx0, res_hc, res_hc->ne[0], hc, n_tokens);
    }
    cb(res_hc, "mtp_eh_proj", il);

    ggml_tensor * inject = nullptr;
    ggml_tensor * cur = build_hc_mix(res_hc,
            layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
            &inject, il);
    cb(cur, "mtp_hc_attn_pre", il);

    if (mtp_qsa) {
        // the trunk's full-attention layer as-is (QSA top-k from this block's indexer, then sparse attention)
        cur = build_layer_attn(inp_attn, mctx_hyb, cur, inp_pos, sections, il);
        cb(cur, "mtp_attn_out", il);
    } else {
        // ---- dense attention, mirroring the trunk's full-attention branch ----
        const int64_t n_embd_head = hparams.n_embd_head_v();
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

        ggml_tensor * Qcur_full = build_lora_mm(layer.wq, cur, layer.wq_s);
        cb(Qcur_full, "mtp_Qcur_full", il);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "mtp_Qcur_normed", il);

        ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
            ggml_element_size(Qcur_full) * n_embd_head * 2,
            ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
            ggml_element_size(Qcur_full) * n_embd_head);
        gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
        cb(gate, "mtp_gate", il);

        ggml_tensor * Kcur = build_lora_mm(layer.wk, cur, layer.wk_s);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Kcur = build_norm(Kcur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        cb(Kcur, "mtp_Kcur_normed", il);

        ggml_tensor * Vcur = build_lora_mm(layer.wv, cur, layer.wv_s);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        cb(Vcur, "mtp_Vcur", il);

        // IMRoPE, same convention and freq_base as the trunk
        Qcur = ggml_rope_multi(ctx0, Qcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_multi(ctx0, Kcur, inp_pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "mtp_Qcur", il);
        cb(Kcur, "mtp_Kcur", il);

        const float kq_scale = hparams.f_attention_scale == 0.0f
                ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        cur = build_attn(inp_attn,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "mtp_attn_pregate", il);

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, gate));
        cb(cur, "mtp_attn_gated", il);

        cur = build_lora_mm(layer.wo, cur, layer.wo_s);
        cb(cur, "mtp_attn_out", il);
    }

    if (inp_out_ids) {
        cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
        inject = ggml_get_rows(ctx0, inject, inp_out_ids);

        res_hc = ggml_reshape_2d(ctx0, res_hc, hc_dim, res_hc->ne[2]);
        res_hc = ggml_get_rows(ctx0, res_hc, inp_out_ids);
        res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, res_hc->ne[1]);
    }

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_attn_post", il);

    // ---- MoE, identical to the trunk's build_layer_ffn ----
    cur = build_hc_mix(res_hc,
            layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
            &inject, il);
    cb(cur, "mtp_hc_ffn_pre", il);

    cur = build_layer_ffn(cur, il);
    cb(cur, "mtp_ffn_out", il);

    res_hc = build_hc_combine(res_hc, cur, inject, il);
    cb(res_hc, "mtp_hc_ffn_post", il);

    // The next draft step re-enters here, so export the wide stream before it is collapsed.
    // As in the trunk, export the combine result rather than a reshape view of it.
    cb(res_hc, "h_nextn", -1);
    res->t_h_nextn = res_hc;

    // the head's own mixer collapses the streams and doubles as the output norm
    cur = build_hc_mix(res_hc,
            layer.nextn.hc_head_norm, layer.nextn.hc_head_down, layer.nextn.hc_head_up,
            nullptr, nullptr, -1);
    cb(cur, "mtp_hc_head", -1);

    // deliberately no res->t_embd: it would be n_embd wide while the context sizes its
    // embedding buffer by n_embd_out (the wide stream). The driver reads t_h_nextn instead.

    ggml_tensor * head_w = layer.nextn.shared_head_head ? layer.nextn.shared_head_head : model.output;
    ggml_tensor * head_s = layer.nextn.shared_head_head ? layer.nextn.shared_head_head_s : model.output_s;
    GGML_ASSERT(head_w && "QWEN4EXP MTP: missing LM head (nextn.shared_head_head or model.output)");

    cur = build_lora_mm(head_w, cur, head_s);
    if (model.d2t) {
        // reduced-vocabulary head: scatter its rows into a full-vocabulary row at -inf, as EAGLE-3 does, so the
        // samplers and the verification see ordinary logits (tokens outside the subset are never drafted)
        const int64_t n_rows    = cur->ne[0];
        const int64_t n_outputs = cur->ne[1];
        const int64_t n_vocab   = (int64_t) model.vocab.n_tokens();
        GGML_ASSERT(model.d2t->type == GGML_TYPE_I64 && model.d2t->ne[0] == n_rows);
        ggml_tensor * logits = ggml_fill(ctx0, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, n_vocab, n_outputs), -INFINITY);
        cur = ggml_set_rows(ctx0, logits,
                ggml_reshape_3d(ctx0, cur,       1,      n_rows, n_outputs),
                ggml_reshape_3d(ctx0, model.d2t, n_rows, 1,      1));
        cur = ggml_reshape_2d(ctx0, cur, n_vocab, n_outputs);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_qwen4exp::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_qwen4exp::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    // the one numerical difference from Qwen3.5's GDN: sigmoid output gate, not silu
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated = ggml_sigmoid(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated);
}

// QSA attends to a budget of whole blocks of compress_ratio tokens, plus the incomplete tail
// one mean-pooled indexer key scores each block; set_input resolves the cache layout
class llama_model_qwen4exp::llm_graph_input_qsa : public llm_graph_input_i {
public:
    llm_graph_input_qsa(const llama_memory_hybrid_idx_context * mctx, uint32_t ratio, bool blk_bias, bool scores, uint32_t top_k, bool causal_attn) :
        mctx(mctx), ratio(ratio), blk_bias(blk_bias), scores(scores), top_k(top_k), causal_attn(causal_attn) {}

    // halo-hybrid: the scores (and their inputs) are only needed when the KV window is wider than top-k returns
    static bool need_scores(int64_t n_kv, uint32_t ratio, uint32_t top_k) {
        static const bool always = getenv("LLAMA_QSA_ALWAYS_SCORE") != nullptr;
        return always || n_kv > (int64_t) top_k + (int64_t) ratio - 1;
    }
    virtual ~llm_graph_input_qsa() = default;

    // halo-hybrid (C6): 0 = no block-key cache, 1 = full re-pool that refreshes the cache, 2 = recompute only the
    // n_re trailing blocks [first, n_bid) and score against the cache
    // several streams: every stream of the graph (n_stream) must hold one planned sequence
    static int blk_mode_for(const llama_memory_hybrid_idx_context * mctx, const llama_ubatch & ub, uint32_t ratio, bool scores,
            int32_t n_re, int64_t n_stream, bool shared, llama_memory_hybrid_idx::qsa_plan & plan) {
        plan = {};
        const llama_memory_hybrid_idx * mem = mctx ? mctx->get_mem() : nullptr;
        if (!scores || mem == nullptr) {
            return 0;
        }
        plan = mem->qsa_blk_plan(ub, ratio, n_re, mctx->get_idx()->get_kv_lo(), mctx->get_idx()->get_n_kv());
        if (!plan.usable || plan.shared != shared || (int64_t) plan.n_bid.size() != n_stream) {
            return 0;
        }
        return plan.incr ? 2 : 1;
    }

    // halo-hybrid: a shared pool (one unified cache holding several sequences, attended through the KV window) whose
    // layout plan holds: the graph's QSA groups are the ubatch's sequences, each scored against its own slab of the
    // block-key cache and its cells found from its base (ggml_qsa_top_k with cell bases). Needs the block-level top-k.
    // LLAMA_QSA_SHARED=0: one group over the whole window, as without the plan
    static bool shared_for(const llama_memory_hybrid_idx_context * mctx, const llama_ubatch & ub, uint32_t ratio, bool scores,
            bool causal_attn, llama_memory_hybrid_idx::qsa_plan & plan) {
        static const bool enabled = getenv("LLAMA_QSA_SHARED") == nullptr || atoi(getenv("LLAMA_QSA_SHARED")) != 0;
        static const bool topk    = getenv("LLAMA_QSA_BLOCK_TOPK") == nullptr || atoi(getenv("LLAMA_QSA_BLOCK_TOPK")) != 0;
        plan = {};
        const llama_memory_hybrid_idx * mem = mctx ? mctx->get_mem() : nullptr;
        if (!enabled || !topk || !scores || !causal_attn || mem == nullptr || !mem->qsa_shared_pool()) {
            return false;
        }
        plan = mem->qsa_identity(ub, ratio, mctx->get_idx()->get_kv_lo(), mctx->get_idx()->get_n_kv());
        return plan.usable && plan.shared;
    }
    static int32_t n_re_for(uint32_t n_tokens, uint32_t ratio) {
        return (int32_t) ((n_tokens + ratio - 1)/ratio) + 1;
    }

    void set_input(const llama_ubatch * ubatch) override {
        mctx->get_idx()->set_input_k_idxs(k_idxs, ubatch);
        // the incremental graph (blk_mode 2) does not read blk_cells / blk_pos, so the scheduler leaves them unallocated:
        // fill host shadows instead (the re_* inputs are taken from them below)
        ggml_tensor bc_shadow, bp_shadow, cb_shadow;
        ggml_tensor * bc = blk_cells;
        ggml_tensor * bp = blk_pos;
        ggml_tensor * cbk = cell_blk;
        if (scores && cell_blk->data == nullptr) {   // the block-level top-k (C4) does not read it either
            cell_blk_host.resize(ggml_nelements(cell_blk));
            cb_shadow = *cell_blk; cb_shadow.data = cell_blk_host.data(); cbk = &cb_shadow;
        }
        if (scores && blk_cells->data == nullptr) {
            blk_cells_host.resize(ggml_nelements(blk_cells));
            bc_shadow = *blk_cells; bc_shadow.data = blk_cells_host.data(); bc = &bc_shadow;
        }
        if (scores && blk_pos->data == nullptr) {
            blk_pos_host.resize(ggml_nelements(blk_pos));
            bp_shadow = *blk_pos; bp_shadow.data = blk_pos_host.data(); bp = &bp_shadow;
        }
        if (scores) {
            mctx->set_input_qsa(cbk, bc, bp, bias, ubatch, ratio, blk_bias, causal_attn);
        }
        if (blk_topk) {
            llama_memory_hybrid_idx::qsa_plan tp;
            const int64_t ng = shared ? ggml_nelements(n_bid_t)/2 : ggml_nelements(n_bid_t);
            GGML_ASSERT(blk_topk_for(mctx, *ubatch, ratio, scores, blk_bias, ng, shared, tp) &&
                    "qsa block top-k: the layout changed between graph build and set_input");
            GGML_ASSERT(ggml_backend_buffer_is_host(n_bid_t->buffer));
            std::copy(tp.n_bid.begin(), tp.n_bid.end(), (int32_t *) n_bid_t->data);
            if (shared) {
                std::copy(tp.base.begin(), tp.base.end(), (int32_t *) n_bid_t->data + ng);
            }
        }
        if (blk_mode == 0) {
            return;
        }
        const int64_t ns = cell_blk->ne[1];
        llama_memory_hybrid_idx::qsa_plan plan;
        const int mode = blk_mode_for(mctx, *ubatch, ratio, scores, n_re, ns, shared, plan);
        GGML_ASSERT(mode == blk_mode && plan.s0 == blk_s0 && "qsa block-key cache: the plan changed between graph build and set_input");
        if (blk_mode == 2) {
            // hole-free single sequence per stream: block id == position / ratio == the bid set_input_qsa numbered, so the
            // cells of block b of stream s are row b of stream s's blk_cells
            GGML_ASSERT(ggml_backend_buffer_is_host(re_cells->buffer) && bc->data != nullptr);
            const int64_t n_blocks = bc->ne[0] / ratio;
            const int32_t * bcd = (const int32_t *) bc->data;
            int32_t * rc = (int32_t *) re_cells->data;
            int64_t * ri = (int64_t *) re_ids->data;
            int32_t * rp = (int32_t *) re_pos->data;
            for (int64_t st = 0; st < ns; ++st) {
                for (int32_t k = 0; k < n_re; ++k) {
                    const int32_t b = plan.first[st] + k;
                    for (uint32_t m = 0; m < ratio; ++m) {
                        rc[(st*n_re + k)*ratio + m] = bcd[(st*n_blocks + b)*ratio + m];
                    }
                    ri[st*n_re + k] = b;
                    for (int sec = 0; sec < 4; ++sec) {
                        rp[sec*n_re*ns + st*n_re + k] = b*(int32_t) ratio;
                    }
                }
            }
        }
        // after this graph the cache holds every complete block of each stream
        const auto * mem = mctx->get_mem();
        for (int64_t st = 0; st < ns; ++st) {
            mem->qsa_valid_pos[plan.s0 + st] = (llama_pos) plan.n_bid[st] * (llama_pos) ratio;
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx);

        const auto * idx = mctx->get_idx();
        if (idx == nullptr) {
            return false;
        }

        const int64_t n_kv     = idx->get_n_kv();
        const int64_t n_kvs    = mctx->get_n_stream();

        bool res = true;

        res &= params.ubatch.n_tokens % n_kvs == 0;

        res &= k_idxs->ne[0]    == params.ubatch.n_tokens;
        res &= need_scores(n_kv, ratio, top_k) == scores;
        if (!scores) {
            return res;
        }

        llama_memory_hybrid_idx::qsa_plan sp;
        const bool shared_now = shared_for(mctx, params.ubatch, ratio, scores, causal_attn, sp);
        res &= shared_now == shared;
        if (!res) {
            return false;
        }
        const int64_t n_stream = shared ? (int64_t) sp.n_bid.size() : n_kvs;   // the QSA groups
        const int64_t n_blocks = shared ? sp.n_blocks : (n_kv + ratio - 1)/ratio;
        res &= params.ubatch.n_tokens % n_stream == 0;
        res &= cell_blk->ne[0]  == n_kv;
        res &= cell_blk->ne[1]  == n_stream;
        res &= blk_cells->ne[0] == (int64_t) ratio*n_blocks;
        res &= blk_pos->ne[0]   == 4*n_blocks*n_stream;
        res &= bias->ne[0]      == (blk_bias ? n_blocks : n_kv);
        res &= bias->ne[1]      == params.ubatch.n_tokens/n_stream;

        llama_memory_hybrid_idx::qsa_plan plan;
        const int mode_now = blk_mode_for(mctx, params.ubatch, ratio, scores, n_re_for(params.ubatch.n_tokens / n_stream, ratio), n_stream, shared, plan);
        {
            static int n_dbg = getenv("LLAMA_QSA_BLK_DEBUG") ? 40 : 0;
            if (n_dbg > 0 && (mode_now != blk_mode || !res)) {
                n_dbg--;
                LLAMA_LOG_WARN("qsa-blk can_reuse: n_tokens %u mode built %d now %d, other checks %d, n_bid %d first %d\n",
                    params.ubatch.n_tokens, blk_mode, mode_now, (int) res, plan.n_bid.empty() ? 0 : plan.n_bid[0],
                    plan.first.empty() ? 0 : plan.first[0]);
            }
        }
        res &= mode_now == blk_mode;
        res &= mode_now == 0 || plan.s0 == blk_s0;
        res &= n_re == n_re_for(params.ubatch.n_tokens / n_stream, ratio);
        llama_memory_hybrid_idx::qsa_plan tp;
        res &= blk_topk_for(mctx, params.ubatch, ratio, scores, blk_bias, n_stream, shared, tp) == blk_topk;

        return res;
    }

    // per stream: a cell index names a different token in each stream
    ggml_tensor * k_idxs    = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_blk  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * blk_cells = nullptr;   // I32 [ratio*n_blocks, n_stream]
    ggml_tensor * blk_pos   = nullptr;   // I32 [4*n_blocks*n_stream]
    ggml_tensor * bias      = nullptr;   // F32 [n_blocks or n_kv, n_tokens/n_stream, n_stream]

    const llama_memory_hybrid_idx_context * mctx;
    const uint32_t ratio;

    // the per-cell half of the bias is the attention mask, so only the per-block half is uploaded
    const bool blk_bias;
    const bool scores;       // halo-hybrid: false = keys only (top-k would return every cell)
    const uint32_t top_k;

    // halo-hybrid (C6): block-key cache mode and, for mode 2, the blocks to recompute
    int      blk_mode = 0;
    int32_t  n_re     = 0;              // recomputed blocks per stream
    uint32_t blk_s0   = 0;              // physical stream of the graph's first stream (the cache rows it views)
    ggml_tensor * re_cells = nullptr;   // I32 [ratio*n_re, n_stream]  cells of each recomputed block
    ggml_tensor * re_ids   = nullptr;   // I64 [n_re, n_stream]        their block ids (rows of the stream's cache)
    ggml_tensor * re_pos   = nullptr;   // I32 [4*n_re*n_stream]       mrope position rows of their first token

    std::vector<int32_t> blk_cells_host, blk_pos_host, cell_blk_host;   // shadows when the graph does not allocate them

    // halo-hybrid (C4): block-level top-k (ggml_qsa_top_k) instead of expanding the scores to every cell
    bool          blk_topk  = false;
    ggml_tensor * n_bid_t   = nullptr;   // I32 [n_stream], [2*n_stream] with the groups' base cells when shared
    static bool blk_topk_for(const llama_memory_hybrid_idx_context * mctx, const llama_ubatch & ub, uint32_t ratio, bool scores,
            bool blk_bias, int64_t n_stream, bool shared, llama_memory_hybrid_idx::qsa_plan & plan) {
        static const bool enabled = getenv("LLAMA_QSA_BLOCK_TOPK") == nullptr || atoi(getenv("LLAMA_QSA_BLOCK_TOPK")) != 0;
        plan = {};
        if (!enabled || !scores || !blk_bias || !mctx || !mctx->get_mem()) {
            return false;
        }
        plan = mctx->get_mem()->qsa_identity(ub, ratio, mctx->get_idx()->get_kv_lo(), mctx->get_idx()->get_n_kv());
        return plan.usable && plan.shared == shared && (int64_t) plan.n_bid.size() == n_stream;
    }

    // halo-hybrid: the graph's QSA groups are the sequences of a shared pool (shared_for)
    bool shared = false;

    // this is fixed for the graph's lifetime, as causal_attn is part of the reuse key (llm_graph_params::allow_reuse)
    const bool causal_attn;
};

ggml_tensor * llama_model_qwen4exp::graph::build_qsa_top_k(
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *                           cur,
        ggml_tensor *                           inp_pos,
        ggml_tensor *                           kq_mask,
        int *                                   sections,
        int                                     il) {
    const llama_kv_cache_context * mctx_idx = mctx_hyb->get_idx();

    const int64_t idx_dim  = hparams.indexer_head_size;
    const int64_t n_idx_h  = hparams.indexer_n_head;
    const int64_t r        = hparams.dsv4_compress_ratios[il];
    const int64_t n_kv     = mctx_idx->get_n_kv();

    GGML_ASSERT(r > 0);

    // build_attn_qsa and the KQ mask need the tokens to divide evenly across the streams
    const int64_t n_kvs = mctx_hyb->get_n_stream();   // streams of the K/V views and the KQ mask
    GGML_ASSERT(n_tokens % n_kvs == 0);

    // halo-hybrid: the indexer's groups - the KV streams, or the sequences of a shared pool (shared_for); "stream" below
    // means such a group
    llama_memory_hybrid_idx::qsa_plan sp;
    const bool shared = llm_graph_input_qsa::shared_for(mctx_hyb, ubatch, (uint32_t) r,
            llm_graph_input_qsa::need_scores(n_kv, (uint32_t) r, hparams.indexer_top_k), cparams.causal_attn, sp);
    const int64_t n_stream = shared ? (int64_t) sp.n_bid.size() : n_kvs;
    const int64_t n_blocks = shared ? sp.n_blocks : (n_kv + r - 1)/r;
    GGML_ASSERT(n_tokens % n_stream == 0);
    const int64_t n_tps = n_tokens/n_stream;

    // only the "which block is visible" half of the bias varies per block
    // the rest is the visible/not test the attention mask already carries, so upload the per-block half only: 1/ratio of the cells
    // alibi writes distances instead of a mask, so it opts out
    // the mask also holds an mrope rule for the query's own position, but only 2d image positions can differ there
    // halo-hybrid: non-causal graphs keep the per-cell bias - the block-key cache (C6) and the block-level top-k (C4)
    // assume the causal rule; set_causal_attn therefore still re-reserves (llama-context)
    const bool blk_bias = kq_mask != nullptr &&
        kq_mask->ne[0] == n_kv && kq_mask->ne[1] == n_tokens/n_kvs && kq_mask->ne[3] == n_kvs &&
        cparams.causal_attn && !hparams.use_alibi;

    // nothing above depends on the layer, so the layers sharing a ratio share one input set
    llm_graph_input_qsa * inp = nullptr;

    const auto it = qsa_inps.find((uint32_t) r);
    if (it != qsa_inps.end()) {
        inp = it->second;
    } else {
        const bool scores = llm_graph_input_qsa::need_scores(n_kv, (uint32_t) r, hparams.indexer_top_k);
        auto qsa = std::make_unique<llm_graph_input_qsa>(mctx_hyb, (uint32_t) r, blk_bias, scores, hparams.indexer_top_k, cparams.causal_attn);
        qsa->shared = shared;

        qsa->k_idxs    = mctx_idx->build_input_k_idxs(ctx0, ubatch);
        if (scores) {
            qsa->cell_blk  = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_kv, n_stream);
            qsa->blk_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*n_blocks, n_stream);
            qsa->blk_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*n_blocks*n_stream);
            qsa->bias      = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, blk_bias ? n_blocks : n_kv, n_tps, n_stream);

            ggml_set_input(qsa->cell_blk);
            ggml_set_input(qsa->blk_cells);
            ggml_set_input(qsa->blk_pos);
            ggml_set_input(qsa->bias);

            llama_memory_hybrid_idx::qsa_plan plan;
            qsa->n_re     = llm_graph_input_qsa::n_re_for(n_tps, (uint32_t) r);
            qsa->blk_mode = mctx_hyb->get_mem()->get_qsa_blk_k(il) != nullptr
                ? llm_graph_input_qsa::blk_mode_for(mctx_hyb, ubatch, (uint32_t) r, scores, qsa->n_re, n_stream, shared, plan) : 0;
            qsa->blk_s0   = plan.s0;
            {
                llama_memory_hybrid_idx::qsa_plan tp;
                qsa->blk_topk = llm_graph_input_qsa::blk_topk_for(mctx_hyb, ubatch, (uint32_t) r, scores, blk_bias, n_stream, shared, tp);
                // a shared pool's groups only exist for the block-level top-k (cells from each group's base)
                GGML_ASSERT(!shared || qsa->blk_topk);
                if (qsa->blk_topk) {
                    qsa->n_bid_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, shared ? 2*n_stream : n_stream);
                    ggml_set_input(qsa->n_bid_t);
                }
            }
            {
                static int n_dbg = getenv("LLAMA_QSA_BLK_DEBUG") ? 40 : 0;
                if (n_dbg > 0) {
                    n_dbg--;
                    LLAMA_LOG_WARN("qsa-blk build: n_tokens %u n_stream %lld n_kv %lld mode %d topk %d s0 %u n_bid[0] %d first[0] %d\n",
                        ubatch.n_tokens, (long long) n_stream, (long long) n_kv, qsa->blk_mode, (int) qsa->blk_topk, plan.s0,
                        plan.n_bid.empty() ? 0 : plan.n_bid[0], plan.first.empty() ? 0 : plan.first[0]);
                }
            }
            if (qsa->blk_mode == 2) {
                qsa->re_cells = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, r*qsa->n_re, n_stream);
                qsa->re_ids   = ggml_new_tensor_2d(ctx0, GGML_TYPE_I64, qsa->n_re, n_stream);
                qsa->re_pos   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 4*qsa->n_re*n_stream);
                ggml_set_input(qsa->re_cells);
                ggml_set_input(qsa->re_ids);
                ggml_set_input(qsa->re_pos);
            }
        }

        inp = qsa.get();
        res->add_input(std::move(qsa));
        qsa_inps.emplace((uint32_t) r, inp);
    }

    // cached indexer keys are raw: pooling precedes norm and rotation, so apply neither
    ggml_tensor * k_raw = build_lora_mm(model.layers[il].index_k_proj, cur);
    k_raw = ggml_reshape_3d(ctx0, k_raw, idx_dim, 1, n_tokens);
    cb(k_raw, "indexer_k_raw", il);

    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, k_raw, inp->k_idxs, il));

    // halo-hybrid: with the whole window inside the budget, top-k returns every cell and the rebuilt mask equals the
    // KQ mask (masked cells stay -inf through the added mask): the keys still go into the cache (a later, wider window
    // pools them), the scoring is skipped and the caller attends densely. LLAMA_QSA_ALWAYS_SCORE=1 restores it.
    if (!inp->scores) {
        return nullptr;
    }

    // one key head, so rows are contiguous. get_k gives [idx_dim, n_head_kv, n_kv, n_kvs].
    ggml_tensor * k_all = mctx_idx->get_k(ctx0, il);
    k_all = ggml_view_3d(ctx0, k_all, idx_dim, n_kv, n_kvs, k_all->nb[2], k_all->nb[3], 0);

    // pool (mean of the r member keys), norm and rotate nb blocks whose member cells are `cells` [r*nb, ns]
    auto pool_blocks = [&](ggml_tensor * cells, ggml_tensor * pos, int64_t nb, int64_t ns) {
        // gathers per stream: blk_cells row s indexes stream s's own cells; the groups of a shared pool index the one
        // window, so their rows are one list
        ggml_tensor * members = ggml_get_rows(ctx0, k_all, k_all->ne[2] == ns ? cells : ggml_reshape_1d(ctx0, cells, ggml_nelements(cells)));
        members = ggml_reshape_4d(ctx0, members, idx_dim, r, nb, ns);

        // mean over the block members; r is small, so summing slices beats a transpose plus sum_rows
        ggml_tensor * pooled = nullptr;
        for (int64_t i = 0; i < r; ++i) {
            ggml_tensor * slice = ggml_cont(ctx0,
                    ggml_view_3d(ctx0, members, idx_dim, nb, ns,
                            members->nb[2], members->nb[3], i*members->nb[1]));
            pooled = pooled ? ggml_add(ctx0, pooled, slice) : slice;
        }
        pooled = ggml_scale(ctx0, pooled, 1.0f/(float) r);
        cb(pooled, "indexer_k_pooled", il);

        // count blocks along ne1: rms_norm launches gridDim.y = ne2, capped at 65535, and 262144/4 = 65536
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, nb*ns, 1);
        pooled = build_norm(pooled, model.layers[il].index_k_norm, nullptr, LLM_NORM_RMS, il);

        // rope wants [n_dims, n_head, n_tokens]: lay every stream's blocks flat, split after.
        pooled = ggml_reshape_3d(ctx0, pooled, idx_dim, 1, nb*ns);
        pooled = ggml_rope_multi(ctx0, pooled, pos, nullptr,
                n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        return ggml_reshape_3d(ctx0, pooled, idx_dim, nb, ns);
    };

    ggml_tensor * pooled = nullptr;
    // halo-hybrid (C6): the finished block keys persist across graphs (llama_memory_hybrid_idx::get_qsa_blk_k); a
    // decode step recomputes only the trailing blocks its tokens touch. Every step is per block row (pool, norm, rope),
    // so the cached rows equal the ones a full re-pool computes and the scores are bit-identical.
    ggml_tensor * blk_k = inp->blk_mode != 0 ? mctx_hyb->get_mem()->get_qsa_blk_k(il) : nullptr;
    if (blk_k != nullptr) {
        // the graph's streams are the consecutive physical streams from blk_s0 (qsa_blk_plan checked it)
        blk_k = ggml_view_3d(ctx0, blk_k, idx_dim, blk_k->ne[1], n_stream, blk_k->nb[1], blk_k->nb[2], inp->blk_s0*blk_k->nb[2]);
    }
    if (blk_k != nullptr && inp->blk_mode == 2) {
        ggml_tensor * fresh = pool_blocks(inp->re_cells, inp->re_pos, inp->n_re, n_stream);
        ggml_tensor * upd = ggml_set_rows(ctx0, blk_k, fresh, inp->re_ids);
        pooled = ggml_view_3d(ctx0, upd, idx_dim, n_blocks, n_stream, upd->nb[1], upd->nb[2], 0);
    } else {
        pooled = pool_blocks(inp->blk_cells, inp->blk_pos, n_blocks, n_stream);
        if (blk_k != nullptr) {
            // refresh the cache for the steps that follow (rows past the complete blocks are never read as valid)
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, pooled,
                    ggml_view_3d(ctx0, blk_k, idx_dim, n_blocks, n_stream, blk_k->nb[1], blk_k->nb[2], 0)));
        }
    }
    cb(pooled, "indexer_k", il);

    ggml_tensor * q = build_lora_mm(model.layers[il].index_q_proj, cur);
    q = ggml_reshape_3d(ctx0, q, idx_dim, n_idx_h, n_tokens);
    q = build_norm(q, model.layers[il].index_q_norm, nullptr, LLM_NORM_RMS, il);
    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);
    cb(q, "indexer_q", il);

    // the reference returns indexer_top_k + compress_ratio - 1: whole blocks plus the tail
    const int64_t width = std::min<int64_t>(n_kv, (int64_t) hparams.indexer_top_k + r - 1);

    // the head-summed block scores [n_blocks, n_c, n_stream] of the queries [c0, c0 + n_c) of every stream
    // halo-hybrid: GGML_OP_QSA_HEAD_SUM does the relu, the head sum and the bias add in one pass (the separate ops made
    // ~4x the passes over [n_blocks, n_tokens] floats, the largest depth-growing elementwise cost). LLAMA_QSA_HEAD_SUM=0
    // restores the separate ops. bias: added in the same pass when given (the caller then skips its own add)
    static const bool fused_head_sum = getenv("LLAMA_QSA_HEAD_SUM") == nullptr || atoi(getenv("LLAMA_QSA_HEAD_SUM")) != 0;
    auto score_heads = [&](int64_t c0, int64_t n_c, ggml_tensor * bias) -> ggml_tensor * {
        ggml_tensor * q_c = c0 == 0 && n_c == n_tps ? q : ggml_view_3d(ctx0, q, idx_dim, n_idx_h, n_c, q->nb[1], q->nb[2], c0*q->nb[2]);

        // rectify each head dot product before the sum, as in the DeepSeek lightning indexer
        // mul_mat matches ne[2], so the queries of stream s only meet the blocks of stream s
        ggml_tensor * score = ggml_mul_mat(ctx0, pooled,
                ggml_reshape_3d(ctx0, q_c, idx_dim, n_idx_h*n_c, n_stream));
        score = ggml_reshape_4d(ctx0, score, n_blocks, n_idx_h, n_c, n_stream);
        if (fused_head_sum) {
            return ggml_qsa_head_sum(ctx0, score, bias);
        }
        score = ggml_relu_inplace(ctx0, score);

        // the heads sit side by side on ne[1] and there are only a few of them
        ggml_tensor * summed = nullptr;
        for (int64_t h = 0; h < n_idx_h; ++h) {
            ggml_tensor * slice = ggml_view_3d(ctx0, score, n_blocks, n_c, n_stream,
                    score->nb[2], score->nb[3], h*score->nb[1]);
            summed = summed ? ggml_add(ctx0, summed, slice) : ggml_cont(ctx0, slice);
        }
        return bias ? ggml_add(ctx0, summed, bias) : summed;
    };

    // every token of a block gets the block score; the budget is whole blocks, so top-k cuts on a block boundary.
    // score: [n_blocks, n_c, n_stream] biased block scores of the queries [c0, c0 + n_c)
    auto expand_top_k = [&](ggml_tensor * score, int64_t c0, int64_t n_c) -> ggml_tensor * {
        const bool whole = c0 == 0 && n_c == n_tps;
        ggml_tensor * expanded = ggml_get_rows(ctx0,
                ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3)), inp->cell_blk);
        expanded = ggml_cont(ctx0, ggml_permute(ctx0, expanded, 1, 0, 2, 3));

        if (blk_bias) {
            // flash attention keeps the mask in f16; the scores are f32
            ggml_tensor * mask = whole ? kq_mask : ggml_view_2d(ctx0, kq_mask, n_kv, n_c, kq_mask->nb[1], c0*kq_mask->nb[1]);
            mask = mask->type == GGML_TYPE_F32 ? mask : ggml_cast(ctx0, mask, GGML_TYPE_F32);
            expanded = ggml_add(ctx0, expanded, ggml_reshape_3d(ctx0, mask, n_kv, n_c, n_stream));
        } else {
            ggml_tensor * bias = whole ? inp->bias : ggml_view_2d(ctx0, inp->bias, n_kv, n_c, inp->bias->nb[1], c0*inp->bias->nb[1]);
            expanded = ggml_add(ctx0, expanded, bias);
        }
        cb(expanded, "indexer_score_tokens", il);

        return ggml_cont(ctx0, ggml_top_k(ctx0, expanded, width));
    };

    // halo-hybrid: at long context the [n_blocks x heads x tokens] scores and the per-cell expansion are each
    // n_kv x n_tokens floats (2.5 GiB at 262K x 2560): build them in chunks of at most LLAMA_QSA_CHUNK_MB (default 2048;
    // chunking costs ~2.5% at 113K, so it only engages where a tensor would be larger - the 262K preset sets 256)
    // per such tensor at the full context (sized from n_ctx, not n_kv, so consecutive ubatches - the two prefill lanes -
    // build the same graph). Chunks only split along the query tokens, so every row is the same computation. One stream
    // only; LLAMA_QSA_CHUNK_MB=0 disables it.
    int64_t n_chunk = n_tps;
    if (n_stream == 1) {
        static const int64_t budget = (getenv("LLAMA_QSA_CHUNK_MB") ? atoll(getenv("LLAMA_QSA_CHUNK_MB")) : 2048) * 1024 * 1024;
        if (budget > 0) {
            const int64_t n_kv_max  = std::max<int64_t>(n_kv, n_ctx);
            const int64_t per_token = std::max<int64_t>(n_kv_max, ((n_kv_max + r - 1)/r)*n_idx_h) * (int64_t) sizeof(float);
            n_chunk = std::max<int64_t>(64, budget / per_token);
            n_chunk = n_chunk >= n_tps ? n_tps : std::max<int64_t>(64, (n_chunk/64)*64);
        }
    }

    ggml_tensor * top_k = nullptr;
    if (n_chunk >= n_tps || inp->blk_topk) {
        // the block scores of all queries: [n_blocks, n_tps, n_stream] is a quarter of an n_kv x n_tokens tensor
        ggml_tensor * score = nullptr;
        // one value per block, so it is cheaper to bias here than after the cells are expanded
        bool biased = false;
        if (n_chunk >= n_tps) {
            score = score_heads(0, n_tps, blk_bias ? inp->bias : nullptr);
            biased = blk_bias;
        } else {
            // chunks write their rows in place; the visibility bias is then added once, from the whole host input
            // (a slice of a host input is a separate synchronous upload per chunk and layer: -13% at 113K)
            // the destination must be a graph node, not a leaf: the allocator keeps leaves for the whole graph, and one
            // per QSA layer stacked up (4 GiB at 128K). A repeat of one element makes it a node freed after this layer.
            for (int64_t c0 = 0; c0 < n_tps; c0 += n_chunk) {
                ggml_tensor * sc = score_heads(c0, std::min(n_chunk, n_tps - c0), nullptr);
                if (score == nullptr) {
                    ggml_tensor * shape = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_blocks, n_tps, n_stream);
                    score = ggml_repeat(ctx0, ggml_view_1d(ctx0, sc, 1, 0), shape);
                }
                score = ggml_set_2d_inplace(ctx0, score, sc, score->nb[1], c0*score->nb[1]);
            }
        }
        cb(score, "indexer_score", il);

        if (blk_bias && !biased) {
            score = ggml_add_inplace(ctx0, score, inp->bias);
        }

        if (inp->blk_topk) {
            // halo-hybrid (C4): one sequence laid out cell j = position j: the cell-level top-k follows from the block
            // scores (ggml_qsa_top_k, exact), without the [n_kv, n_tokens] expansion, its permutes, the mask add and a
            // radix top-k over every cell. LLAMA_QSA_BLOCK_TOPK=0 restores the expansion.
            // several streams: row i of the result belongs to stream i / n_tps, as the ubatch lays its tokens out
            ggml_tensor * qp = ggml_view_1d(ctx0, inp_pos, n_tokens, 0);
            top_k = ggml_qsa_top_k(ctx0, ggml_reshape_3d(ctx0, score, n_blocks, n_tps, n_stream), qp, inp->n_bid_t, (int32_t) width, (int32_t) r);
        } else {
            top_k = expand_top_k(score, 0, n_tps);
        }
    } else {
        // per-cell fallback in chunks: the reserve-sized worst case (it only runs for layouts the block-level top-k
        // cannot take), so it slices its host inputs rather than keeping n_kv x n_tokens device copies
        for (int64_t c0 = 0; c0 < n_tps; c0 += n_chunk) {
            const int64_t n_c = std::min(n_chunk, n_tps - c0);
            ggml_tensor * sc = score_heads(c0, n_c,
                    blk_bias ? ggml_view_2d(ctx0, inp->bias, n_blocks, n_c, inp->bias->nb[1], c0*inp->bias->nb[1]) : nullptr);
            ggml_tensor * tk = expand_top_k(sc, c0, n_c);
            top_k = top_k ? ggml_concat(ctx0, top_k, tk, 1) : tk;
        }
    }

    // build_attn_qsa reads [n_top_k, n_batch, 1, n_kvs], matching the KQ mask (the groups of a shared pool are one
    // KV stream: their rows already index the window)
    top_k = ggml_reshape_4d(ctx0, top_k, width, n_tokens/n_kvs, 1, n_kvs);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

// Dense GQA self-attention restricted to the cells that top_k names.
// The mask build below copies the MLA sparse path in llm_graph_context::build_attn.
ggml_tensor * llama_model_qwen4exp::graph::build_attn_qsa(
        llm_graph_input_attn_kv * inp,
        ggml_tensor *             q_cur,
        ggml_tensor *             k_cur,
        ggml_tensor *             v_cur,
        ggml_tensor *             top_k,
        float                     kq_scale,
        int                       il) {
    // rotate q/k/v before they reach a quantized cache, as the dense path does. the indexer
    // has already scored with its own query in build_qsa_top_k, so top_k is unaffected.
    if (inp->self_k_rot) {
        q_cur = llama_mul_mat_hadamard(ctx0, q_cur, inp->self_k_rot);
        k_cur = llama_mul_mat_hadamard(ctx0, k_cur, inp->self_k_rot);
    }

    if (inp->self_v_rot) {
        v_cur = llama_mul_mat_hadamard(ctx0, v_cur, inp->self_v_rot);
    }

    // these nodes are added to the graph together so that they are not reordered
    // by doing so, the number of splits in the graph is reduced
    // expand k later to enable rope fusion which directly writes into k-v cache
    ggml_build_forward_expand(gf, q_cur);
    ggml_build_forward_expand(gf, v_cur);
    ggml_build_forward_expand(gf, k_cur);

    const auto * mctx_cur = inp->mctx;

    // store to KV cache
    {
        const auto & k_idxs = inp->get_k_idxs();
        const auto & v_idxs = inp->get_v_idxs();

        ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, k_cur, k_idxs, il));
        ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, v_cur, v_idxs, il));
    }

    ggml_tensor * kq_mask = inp->get_kq_mask();

    // prepare new kq mask - starts filled with -INFINITY
    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);

    // reshape KQ mask into tensor with rows of size 1:
    // [n_kv, n_batch, 1, n_stream] -> [1, n_kv, n_batch, n_stream]
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3], kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    // reshape top_k indices: [n_top_k, n_batch, 1, n_stream] -> [n_top_k, n_batch, n_stream, 1]
    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1, top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // prepare zero-filled tensor with rows of size 1: [1, n_top_k, n_batch, n_stream]
    // this will be our source of zero values for unmasking top k mask elements
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    // modify KQ mask by unmasking elements that are in top_k indices
    // ggml_set_rows([1, n_kv, n_batch, n_stream], [1, n_top_k, n_batch, n_stream], [n_top_k, n_batch, n_stream, 1])
    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);

    // reshape to restore the original shape of KQ mask:
    // [1, n_kv, n_batch, n_stream] -> [n_kv, n_batch, 1, n_stream]
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k, kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3], kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    // combine with the original kq mask; in place: the filled mask is this layer's own scratch (saves n_kv x n_tokens
    // halves per lane at long context)
    kq_mask_top_k = ggml_add_inplace(ctx0, kq_mask_top_k, kq_mask);

    ggml_tensor * q = q_cur;
    ggml_tensor * k = mctx_cur->get_k(ctx0, il);
    ggml_tensor * v = mctx_cur->get_v(ctx0, il);

    // Decode fast path (ported from ucicelos/flashnext-hybrid): gather the selected K/V rows and
    // attend over exactly those instead of masking all n_kv cells, so per-token attention is
    // O(n_sel) rather than O(n_kv). Off below LLAMA_QSA_GATHER cells (default 16384, 24576 on an iGPU), where the
    // dense path is cheaper. The value-side rotation is undone after either path identically.
    {
        const int64_t width_qsa = top_k->ne[0];
        const ggml_backend_dev_t dev_attn = model.dev_layer(il);
        const bool igpu = dev_attn != nullptr && ggml_backend_dev_type(dev_attn) == GGML_BACKEND_DEVICE_TYPE_IGPU;
        if (v->nb[1] <= v->nb[2] && attn_top_k_gather_n_sel(k->ne[2], width_qsa, igpu) == width_qsa) {
            ggml_tensor * gcur = build_attn_top_k_gather(kq_mask, k, v, q, top_k, width_qsa, kq_scale, il);
            cb(gcur, "kqv_out", il);
            if (inp->self_v_rot) {
                gcur = llama_mul_mat_hadamard(ctx0, gcur, inp->self_v_rot);
            }
            return gcur;
        }
    }

    // halo-hybrid: the FA op carries the per-query bound (top-k width), which lets the backend walk each query
    // tile's union of selections instead of the whole causal context at prefill (ggml-cuda fattn.cu,
    // shall_use_sparse; the op stays exact, the masked cells are never visited). LLAMA_QSA_SPARSE_FA=0 = dense walk
    static const bool sparse_fa = getenv("LLAMA_QSA_SPARSE_FA") == nullptr || atoi(getenv("LLAMA_QSA_SPARSE_FA")) != 0;
    ggml_tensor * cur = build_attn_mha(q, k, v, nullptr, kq_mask_top_k, nullptr, nullptr,
            sparse_fa ? top_k->ne[0] : 0, kq_scale, il);
    cb(cur, "kqv_out", il);

    // the rotation is its own inverse, so undo it on the value side of the output
    if (inp->self_v_rot) {
        cur = llama_mul_mat_hadamard(ctx0, cur, inp->self_v_rot);
    }

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn(
        llm_graph_input_attn_kv * inp,
        const llama_memory_hybrid_idx_context * mctx_hyb,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // indexer reads the same block input as q/k/v; no cache or no ratio means dense
    const bool qsa = mctx_hyb->get_idx() != nullptr && hparams.dsv4_compress_ratios[il] > 0;

    ggml_tensor * top_k = qsa ? build_qsa_top_k(mctx_hyb, cur, inp_pos, inp->get_kq_mask(), sections, il) : nullptr;

    // Qwen3Next uses a single Q projection that outputs query + gate
    ggml_tensor * Qcur_full = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s); // [ (n_embd_head * 2) * n_head, n_tokens ]
    cb(Qcur_full, "Qcur_full", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
    cb(Kcur, "Kcur", il);

    ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s);
    cb(Vcur, "Vcur", il);

    // pin the three projections of `cur` adjacent in the graph (grouped mul_mat_vec_q quantises
    // the shared input once); the Q norm is expanded later by its consumers
    ggml_build_forward_expand(gf, Qcur_full);
    ggml_build_forward_expand(gf, Kcur);
    ggml_build_forward_expand(gf, Vcur);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply IMRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    if (top_k) {
        cur = build_attn_qsa(inp, Qcur, Kcur, Vcur, top_k, kq_scale, il);
    } else {
        cur = build_attn(inp,
                    nullptr, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    }
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = hparams.ssm_d_state;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);
    GGML_ASSERT(head_v_dim * num_v_heads == d_inner);

    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    // pin the four projections of `cur` -- qkv, z, beta, alpha -- adjacent in the graph so the
    // CUDA backend quantises the shared input once (grouped mul_mat_vec_q)
    ggml_build_forward_expand(gf, qkv_mixed);
    ggml_build_forward_expand(gf, z);
    ggml_build_forward_expand(gf, beta);
    ggml_build_forward_expand(gf, alpha);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];

    // the channels must match how load_arch_tensors sizes wqkv, not ssm_d_inner
    const int64_t conv_channels    = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;

    ggml_tensor * conv_input = build_conv_state_at(inp, conv_states_all, qkv_mixed,
            conv_kernel_size - 1, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, conv_channels);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    // q and k are adjacent in the conv output: one l2_norm over both instead of two launches
    static const bool no_graph_fuse = getenv("LLAMA_NO_GRAPH_FUSE") != nullptr;
    if (!no_graph_fuse) {
        ggml_tensor * qk_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, 2 * num_k_heads, n_seq_tokens, n_seqs,
                ggml_row_size(conv_qkv_mix->type, head_k_dim), nb1_qkv, nb1_qkv * n_seq_tokens, 0);
        qk_conv = ggml_l2_norm(ctx0, qk_conv, eps_norm);
        q_conv = ggml_view_4d(ctx0, qk_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, qk_conv->nb[1], qk_conv->nb[2], qk_conv->nb[3], 0);
        k_conv = ggml_view_4d(ctx0, qk_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs, qk_conv->nb[1], qk_conv->nb[2], qk_conv->nb[3], (size_t) num_k_heads * qk_conv->nb[1]);
    } else {
        q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
        k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);
    }

    // repeat to match shapes when head keys != value keys; unneeded with the fused GDN
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // gated normalization, as self.norm(core_attn_out, z) in the reference
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    ggml_tensor * final_output = ggml_reshape_3d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens, n_seqs);
    cb(final_output, "final_output", il);

    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_seq_tokens * n_seqs);

    return cur;
}

ggml_tensor * llama_model_qwen4exp::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    GGML_ASSERT(model.layers[il].ffn_gate_inp != nullptr);

    // halo-hybrid (P4): when the routed experts live on another device than the layer (APU + card), send them the
    // input as f16 (half the host-link bytes; the F16 expert path converts to f16 anyway): router logits from the f32
    // input on the layer's device, the experts read cast(cast(x, f16), f32) with the f16 cast pinned to the layer's
    // device and the f32 cast to the experts' (llama-context's graph callback). On by default (Swift Q8T hybrid prefill
    // 16K/32K +2.2-2.4%); gated to the ubatches the F16 expert path takes, where it is bit-identical (it fired at >= 256 tokens
    // before, where MMQ read the f16-rounded input: KLD 0.0275 at ub 1024). LLAMA_MOE_F16_CROSS=0 disables.
    ggml_tensor * exp_in = cur;
    ggml_tensor * logits = nullptr;
    {
        static const bool f16_cross = getenv("LLAMA_MOE_F16_CROSS") == nullptr || atoi(getenv("LLAMA_MOE_F16_CROSS")) != 0;
        const ggml_tensor * we = model.layers[il].ffn_gate_exps ? model.layers[il].ffn_gate_exps : model.layers[il].ffn_gate_up_exps;
        ggml_backend_buffer_type_t buft_exp = (we && we->buffer) ? ggml_backend_buffer_get_type(we->buffer) : nullptr;
        ggml_backend_dev_t dev_exp = buft_exp ? ggml_backend_buft_get_device(buft_exp) : nullptr;
        // only where the f16 expert GEMM takes the batch (ggml-cuda mmid-f16 auto rule: >= 40 rows per expert), so the
        // crossing is bit-identical, and only for experts in GPU memory (CPU-resident or host-pinned experts would pay
        // a round trip for nothing)
        const bool exp_gpu = dev_exp && !ggml_backend_buft_is_host(buft_exp) &&
            (ggml_backend_dev_type(dev_exp) == GGML_BACKEND_DEVICE_TYPE_GPU || ggml_backend_dev_type(dev_exp) == GGML_BACKEND_DEVICE_TYPE_IGPU);
        const bool f16_rows = (int64_t) n_tokens*n_expert_used >= 40*(int64_t) n_expert;
        // the cast pair exists for every prefill-sized ubatch (>= 256 tokens) so the two prefill lanes of a pair (e.g.
        // 2560 + 1143 tokens) keep the same split structure (the pair guard runs mismatched lanes one after the other);
        // below the f16 rows threshold it crosses as f32 (exact)
        if (f16_cross && n_tokens >= 256 && exp_gpu && dev_exp != model.dev_layer(il)) {
            logits = build_lora_mm(model.layers[il].ffn_gate_inp, cur);
            cb(logits, "ffn_moe_logits", il);
            ggml_tensor * x16 = ggml_cast(ctx0, cur, f16_rows ? GGML_TYPE_F16 : GGML_TYPE_F32);
            cb(x16, "moe_in_f16", il);
            exp_in = ggml_cast(ctx0, x16, GGML_TYPE_F32);
            cb(exp_in, "moe_in_f32", il);
        }
    }

    ggml_tensor * moe_out =
        build_moe_ffn(exp_in,
            model.layers[il].ffn_gate_inp,
            model.layers[il].ffn_up_exps,
            model.layers[il].ffn_gate_exps,
            model.layers[il].ffn_down_exps,
            nullptr,
            n_expert, n_expert_used,
            LLM_FFN_SILU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il,
            logits, model.layers[il].ffn_gate_up_exps,
            model.layers[il].ffn_up_exps_s,
            model.layers[il].ffn_gate_exps_s,
            model.layers[il].ffn_down_exps_s);
    cb(moe_out, "ffn_moe_out", il);

    // shared experts, as in the Qwen3Next reference
    if (model.layers[il].ffn_up_shexp != nullptr) {
        ggml_tensor * ffn_shexp =
            build_ffn(cur,
                model.layers[il].ffn_up_shexp, NULL, model.layers[il].ffn_up_shexp_s,
                model.layers[il].ffn_gate_shexp, NULL, model.layers[il].ffn_gate_shexp_s,
                model.layers[il].ffn_down_shexp, NULL, model.layers[il].ffn_down_shexp_s,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        // shared expert has its own sigmoided gate (ffn_gate_inp_shexp, one value per token)
        ggml_tensor * shared_gate = build_lora_mm(model.layers[il].ffn_gate_inp_shexp, cur);
        cb(shared_gate, "shared_expert_gate", il);

        shared_gate = ggml_sigmoid(ctx0, shared_gate);
        cb(shared_gate, "shared_expert_gate_sigmoid", il);

        ffn_shexp = ggml_mul(ctx0, ffn_shexp, shared_gate);
        cb(ffn_shexp, "ffn_shexp_gated", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);
    } else {
        cur = moe_out;
    }

    return cur;
}

// PLE n-gram hash embedding: each token gathers ple_n_heads rows of a shared table.
//   mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]);  row = mixed_n % vocab[h] + offset[h]
// The hash runs host-side because ggml has no int64 and no xor. EOS resets the window.

class llm_graph_input_ple : public llm_graph_input_i {
public:
    llm_graph_input_ple(const llama_model_qwen4exp & pmodel,
                        const llama_kv_cache_context * mctx) : pmodel(pmodel), mctx(mctx) {}
    virtual ~llm_graph_input_ple() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override {
        mctx = static_cast<const llama_memory_hybrid_idx_context *>(params.mctx)->get_attn();
        const int64_t n_tokens = params.ubatch.n_tokens;
        if (emb) {
            return emb->ne[1] == n_tokens;
        }
        return rows != nullptr && rows->ne[0] == (int64_t) pmodel.hparams.ple_n_heads * n_tokens;
    }

    ggml_tensor * rows = nullptr;   // I32 [ple_n_heads * n_tokens]
    // When the table lives in a host buffer the gather + dequant is done here
    // instead of by a get_rows node, and fed as F32 [head_dim*n_heads, n_tokens].
    // That keeps the 28 GB table out of the graph, so the scheduler no longer
    // needs a CPU split (and a synchronising host->device copy) every token.
    ggml_tensor * emb = nullptr;

    const llama_model_qwen4exp & pmodel;

    // the predecessor tokens live in the attention KV cells (ext.tok)
    const llama_kv_cache_context * mctx;

    // scratch, reused across set_input() calls
    std::vector<llama_token> prev;
};

// the n-gram rows of every token (ple_n_heads per token); prev holds n_gram - 1 predecessors per token, oldest-first,
// LLAMA_TOKEN_NULL where there is none
static void ple_rows(const llama_hparams & hp, const llama_ubatch & ubatch, const std::vector<llama_token> & prev,
                     std::vector<int32_t> & idx) {
    // an image arrives as an embd batch, so ubatch.token is null, but every position still needs a row for ggml_get_rows
    // stand in the image token id that the reference hashes, or EOS if the file has no such key
    // gemma3n and gemma4 do the same with a hardcoded row 0 of per_layer_token_embd.
    const llama_token img_tok = hp.ple_image_token_id != 0
        ? (llama_token) hp.ple_image_token_id
        : (llama_token) hp.ple_eos_token_id;
    auto tok_of = [&](int64_t k) -> llama_token {
        return ubatch.token ? ubatch.token[k] : img_tok;
    };

    const int64_t n_tokens = ubatch.n_tokens;
    const int64_t n_gram   = hp.ple_ngram_size;
    const int64_t n_heads  = hp.ple_n_heads;
    const int64_t per_gram = hp.ple_heads_per_ngram;
    const int64_t eos      = hp.ple_eos_token_id;
    const int64_t n_prev   = n_gram - 1;

    idx.resize(n_heads * n_tokens);

    std::vector<int64_t> ctx(n_gram);
    for (int64_t i = 0; i < n_tokens; ++i) {
        // an EOS in the window resets everything at or before it
        // a missing predecessor (before the sequence start, or no cached cell) reads as EOS
        // the EOS of the token itself does not cut its own context, as in the reference
        ctx[0] = tok_of(i);
        bool cut = false;
        for (int64_t s = 1; s < n_gram; ++s) {
            // predecessor s positions back; prev[] is oldest-first, missing entries are LLAMA_TOKEN_NULL
            const llama_token t = cut ? LLAMA_TOKEN_NULL : prev[i*n_prev + (n_prev - s)];
            cut = cut || t < 0 || t == eos;
            ctx[s] = cut ? eos : t;
        }

        for (int64_t n = 2; n <= n_gram; ++n) {
            uint64_t mixed = (uint64_t) ctx[0] * hp.ple_layer_multipliers[0];
            for (int64_t j = 1; j < n; ++j) {
                mixed ^= (uint64_t) ctx[j] * hp.ple_layer_multipliers[j];
            }
            const int64_t base = (n - 2) * per_gram;
            for (int64_t g = 0; g < per_gram; ++g) {
                const int64_t h_i = base + g;
                idx[i * n_heads + h_i] =
                    (int32_t) (mixed % hp.ple_head_vocab_sizes[h_i] + hp.ple_head_offsets[h_i]);
            }
        }
    }
}

// gather + dequant the rows from the host (mmapped) table: F32 [head_dim * n_heads, n_tokens]
static void ple_gather(const ggml_tensor * w, int64_t hd, const std::vector<int32_t> & idx, std::vector<float> & out) {
    const auto * tt = ggml_get_type_traits(w->type);   // same dequant the CPU get_rows uses
    out.resize(idx.size() * hd);
    const char * base = (const char *) w->data;
    const size_t row_bytes = ggml_row_size(w->type, hd);
#if defined(__linux__)
    // the table is mmapped and mostly cold: ask for every page this ubatch reads up front, so the kernel reads
    // them in parallel instead of one fault at a time inside the dequant loop (adjacent pages merged per call)
    static const bool willneed = !getenv("LLAMA_PLE_WILLNEED") || atoi(getenv("LLAMA_PLE_WILLNEED")) > 0;
    if (willneed) {
        static const long pg = sysconf(_SC_PAGESIZE);
        std::vector<uintptr_t> pages;
        pages.reserve(idx.size() * 2);
        for (size_t r = 0; r < idx.size(); ++r) {
            const uintptr_t a = (uintptr_t) (base + (size_t) idx[r] * w->nb[1]);
            for (uintptr_t p0 = a & ~(uintptr_t) (pg - 1); p0 < a + row_bytes; p0 += pg) {
                pages.push_back(p0);
            }
        }
        std::sort(pages.begin(), pages.end());
        pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
        for (size_t i = 0; i < pages.size(); ) {
            size_t j = i + 1;
            while (j < pages.size() && pages[j] == pages[j - 1] + pg) { ++j; }
            madvise((void *) pages[i], (j - i) * pg, MADV_WILLNEED);
            i = j;
        }
    }
#endif
    // dequant on a few threads: page faults that the read-ahead has not satisfied yet then overlap
    static const int n_thr_env = getenv("LLAMA_PLE_THREADS") ? atoi(getenv("LLAMA_PLE_THREADS")) : 8;
    const int n_thr = (int) std::max<int64_t>(1, std::min<int64_t>(n_thr_env, (int64_t) idx.size() / 256));
    auto work = [&](int t) {
        const size_t r0 = idx.size() * t / n_thr, r1 = idx.size() * (t + 1) / n_thr;
        for (size_t r = r0; r < r1; ++r) {
            tt->to_float(base + (size_t) idx[r] * w->nb[1], out.data() + r*hd, hd);
        }
    };
    if (n_thr == 1) {
        work(0);
    } else {
        std::vector<std::thread> thr;
        for (int t = 1; t < n_thr; ++t) { thr.emplace_back(work, t); }
        work(0);
        for (auto & th : thr) { th.join(); }
    }
}

// halo-hybrid: prefetched PLE gathers. A prefill ubatch's ~65K scattered rows come mostly from disk (~170 ms per 4096
// tokens); llama_context starts them for both lanes of a pair while the previous pair still runs on the GPUs
// (prefetch_inputs), and set_input takes one only when its rows are exactly the ones it computes.
// LLAMA_PLE_PREFETCH=0 turns it off
namespace {
struct ple_prefetch {
    const llama_model *                 model = nullptr;
    std::vector<int32_t>                idx;
    std::shared_ptr<std::vector<float>> out;
    std::shared_future<void>            done;
};
std::mutex               ple_pf_mu;
std::deque<ple_prefetch> ple_pf;   // the two lanes of a pair, plus stale guesses until they age out
constexpr size_t         ple_pf_max = 4;

// the matching prefetch's rows (waits for its gather), or nullptr; *n_stale: this model's entries that did not match
std::shared_ptr<std::vector<float>> ple_prefetch_take(const llama_model * model, const std::vector<int32_t> & idx, int * n_stale) {
    ple_prefetch hit;
    {
        std::lock_guard<std::mutex> lock(ple_pf_mu);
        *n_stale = 0;
        for (auto it = ple_pf.begin(); it != ple_pf.end(); ++it) {
            if (it->model != model) {
                continue;
            }
            if (it->idx == idx) {
                hit = std::move(*it);
                ple_pf.erase(it);
                break;
            }
            ++*n_stale;
        }
    }
    if (!hit.out) {
        return nullptr;
    }
    hit.done.wait();
    return hit.out;
}

// drops this model's prefetches (all of them with model == nullptr), waiting for gathers still reading its table
void ple_prefetch_drop(const llama_model * model) {
    std::vector<ple_prefetch> dropped;
    {
        std::lock_guard<std::mutex> lock(ple_pf_mu);
        for (auto it = ple_pf.begin(); it != ple_pf.end(); ) {
            if (model == nullptr || it->model == model) {
                dropped.push_back(std::move(*it));
                it = ple_pf.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (auto & e : dropped) {
        e.done.wait();
    }
}
} // namespace

static bool ple_host_gather(const llama_model & model) {
    return model.per_layer_tok_embd != nullptr && model.per_layer_tok_embd->buffer != nullptr &&
        ggml_backend_buffer_is_host(model.per_layer_tok_embd->buffer) && getenv("LLAMA_PLE_GET_ROWS") == nullptr;
}

llama_model_qwen4exp::~llama_model_qwen4exp() {
    ple_prefetch_drop(this);
}

void llama_model_qwen4exp::prefetch_inputs(const llama_ubatch & ubatch, const llama_memory_context_i * mctx, bool pending) const {
    static const bool enabled = !getenv("LLAMA_PLE_PREFETCH") || atoi(getenv("LLAMA_PLE_PREFETCH")) != 0;
    if (!enabled || mctx == nullptr || hparams.ple_n_heads == 0 || ubatch.n_tokens < 64 || !ubatch.token ||
            !ple_host_gather(*this)) {
        return;
    }
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        if (ubatch.n_seq_id[i] != 1) {
            return;   // set_input rejects these
        }
    }

    const auto * kv = static_cast<const llama_memory_hybrid_idx_context *>(mctx)->get_attn();
    const uint32_t n_prev = hparams.ple_ngram_size - 1;
    std::vector<llama_token> prev;
    if (pending) {
        if (!kv->get_prev_tokens_pending(ubatch, n_prev, prev)) {
            return;
        }
    } else {
        kv->get_prev_tokens(ubatch, n_prev, prev);
    }

    ple_prefetch e;
    e.model = this;
    ple_rows(hparams, ubatch, prev, e.idx);
    {
        // lane 1's ubatch is prefetched twice, by lane 0 (pending) and by its own prepare: keep the first
        std::lock_guard<std::mutex> lock(ple_pf_mu);
        for (const auto & q : ple_pf) {
            if (q.model == this && q.idx == e.idx) {
                return;
            }
        }
    }
    e.out = std::make_shared<std::vector<float>>();
    e.done = std::async(std::launch::async,
        [w = per_layer_tok_embd, hd = (int64_t) hparams.ple_head_dim, idx = e.idx, out = e.out] {
            ple_gather(w, hd, idx, *out);
        }).share();

    ple_prefetch old;
    {
        std::lock_guard<std::mutex> lock(ple_pf_mu);
        if (ple_pf.size() >= ple_pf_max) {
            old = std::move(ple_pf.front());
            ple_pf.pop_front();
        }
        ple_pf.push_back(std::move(e));
    }
    if (old.out) {
        old.done.wait();
    }
}

void llm_graph_input_ple::set_input(const llama_ubatch * ubatch) {
    const auto & hp = pmodel.hparams;

    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(mctx != nullptr);

    for (int64_t i = 0; i < n_tokens; ++i) {
        // the preceding tokens would be ambiguous, see get_prev_tokens()
        GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "PLE n-gram embeddings do not support tokens shared by multiple sequences");
    }

    // predecessors come from the KV cells (ext.tok); apply_ubatch() already stored this ubatch, so its own tokens count too
    mctx->get_prev_tokens(*ubatch, hp.ple_ngram_size - 1, prev);

    std::vector<int32_t> idx;
    ple_rows(hp, *ubatch, prev, idx);

    if (emb) {
        int n_stale = 0;
        std::shared_ptr<std::vector<float>> pf = ple_prefetch_take(&pmodel, idx, &n_stale);
        static const int lanes_debug = getenv("LLAMA_LANES_DEBUG") ? atoi(getenv("LLAMA_LANES_DEBUG")) : 0;
        if (lanes_debug >= 3 && n_tokens >= 64) {
            LLAMA_LOG_INFO("ple prefetch: %" PRId64 " tokens %s, %d unmatched\n", n_tokens, pf ? "hit" : "miss", n_stale);
        }
        if (pf) {
            ggml_backend_tensor_set(emb, pf->data(), 0, pf->size()*sizeof(float));
            return;
        }
        std::vector<float> out;
        ple_gather(pmodel.per_layer_tok_embd, hp.ple_head_dim, idx, out);
        ggml_backend_tensor_set(emb, out.data(), 0, out.size()*sizeof(float));
        return;
    }

    ggml_backend_tensor_set(rows, idx.data(), 0, idx.size()*ggml_element_size(rows));
}

// Read a conv history out of its own recurrent row and write the new tail back.
// The shared build_conv_state cannot do this: qwen4exp has two such rows per layer.
ggml_tensor * llama_model_qwen4exp::graph::build_conv_state_at(
        llm_graph_input_rs * inp,
        ggml_tensor *        conv_states_all,
        ggml_tensor *        x,
        int64_t              state_cols,
        int64_t              channels,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const auto kv_head = mctx_cur->get_head();

    const int64_t n_seqs    = ubatch.n_seqs;
    const int64_t row_total = conv_states_all->ne[0];

    // the row is exactly this convolution's state, so the gather is reused as a whole
    GGML_ASSERT(state_cols * channels == row_total);

    auto it = rs_rows.find(conv_states_all);
    if (it == rs_rows.end()) {
        it = rs_rows.emplace(conv_states_all, build_rs(inp, conv_states_all, row_total, n_seqs)).first;
    }
    ggml_tensor * rows = it->second;

    ggml_tensor * state = ggml_reshape_3d(ctx0, rows, state_cols, channels, n_seqs);
    cb(state, "conv_state_at", il);

    ggml_tensor * conv_input = ggml_concat(ctx0, state, ggml_transpose(ctx0, x), 0);

    // [TAG_RECURRENT_ROLLBACK_SPLITS] keep the last state_cols columns once per rollback slot,
    // slot s ending s tokens earlier so a rollback of s tokens reads a history that never saw them
    const size_t row_size = ggml_row_size(conv_states_all->type, row_total);
    const uint32_t mem_size = mctx_cur->get_size();

    const int64_t n_slots = (int64_t) cparams.n_rs_seq + 1;

    for (int64_t slot = 0; slot < n_slots; ++slot) {
        const int64_t s_idx = std::max<int64_t>(0, conv_input->ne[0] - state_cols - slot);

        ggml_tensor * tail = ggml_view_3d(ctx0, conv_input,
                state_cols, channels, n_seqs,
                conv_input->nb[1], conv_input->nb[2],
                ggml_row_size(conv_input->type, s_idx));

        ggml_tensor * dst = ggml_view_2d(ctx0, conv_states_all,
                state_cols * channels, n_seqs,
                conv_states_all->nb[1],
                (slot * mem_size + kv_head) * row_size);

        ggml_build_forward_expand(gf, ggml_cpy(ctx0, tail, dst));
    }

    return conv_input;
}

ggml_tensor * llama_model_qwen4exp::graph::build_inp_ple(
        const llama_memory_hybrid_idx_context * mctx_hyb) {
    const int64_t n_heads = hparams.ple_n_heads;

    // the attention cells see every ubatch regardless of the layer types
    auto ple_inp = std::make_unique<llm_graph_input_ple>(
            static_cast<const llama_model_qwen4exp &>(model), mctx_hyb->get_attn());

    ggml_tensor * emb = nullptr;
    // halo-hybrid: with the 28 GB table in host memory the 16-row gather + dequant runs on the host
    // in set_input and arrives as an F32 input, so the scheduler needs no CPU split for it
    const bool host_gather = ple_host_gather(model);
    if (host_gather) {
        ple_inp->emb = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.ple_head_dim * n_heads, n_tokens);
        ggml_set_input(ple_inp->emb);
        emb = ple_inp->emb;
        res->add_input(std::move(ple_inp));
    } else {
        ple_inp->rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_heads * n_tokens);
        ggml_set_input(ple_inp->rows);
        ggml_tensor * rows = ple_inp->rows;
        res->add_input(std::move(ple_inp));

        // gather then flatten the heads: get_rows lays the head dimension out slowest, as the reference does
        emb = ggml_get_rows(ctx0, model.per_layer_tok_embd, rows);
        emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim * n_heads, n_tokens);
    }
    cb(emb, "ple_embd", -1);

    return emb;
}

ggml_tensor * llama_model_qwen4exp::graph::build_ple(
        llm_graph_input_rs * inp,
        ggml_tensor *        emb,
        ggml_tensor *        hidden,
        int                  il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc * n_embd;

    ggml_tensor * key   = build_lora_mm(model.layers[il].ple_key,   emb);
    ggml_tensor * value = build_lora_mm(model.layers[il].ple_value, emb);

    // both norms group over one hc stream, with a weight over the whole hc*n_embd layout
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a signed square root before the sigmoid
    ggml_tensor * s = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    s = ggml_scale(ctx0, s, 1.0f / sqrtf((float) n_embd));

    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, s), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, s), mag));
    cb(gate, "ple_gate", il);

    // [n_embd, 1, T] value broadcast across the hc streams, scaled by the gate
    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);

    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens),
            model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // depthwise causal conv, dilated by the n-gram size, as a sum of shifted copies
    // ggml_conv_1d_dw is documented as unreliable:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // The history of the earlier ubatches is prepended, so a chunked prefill matches a single-shot one.
    const int64_t kern = hparams.ple_conv_kernel;
    const int64_t dil  = hparams.ple_ngram_size;
    const int64_t hist = (kern - 1) * dil;

    // the conv history is per sequence, so the input carries the sequence axis too
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    // [hist + n_seq_tokens, hc_dim, n_seqs], tokens on ne[0]
    ggml_tensor * padded = build_conv_state_at(inp, inp->mctx->get_p_l(il),
            ggml_reshape_3d(ctx0, normalized, hc_dim, n_seq_tokens, n_seqs),
            hist, hc_dim, il);

    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        // tap k reads (kern-1-k)*dilation positions back
        const int64_t start = hist - (kern - 1 - k) * dil;

        ggml_tensor * shifted = ggml_cont(ctx0,
                ggml_transpose(ctx0,
                        ggml_view_3d(ctx0, padded, n_seq_tokens, hc_dim, n_seqs,
                                padded->nb[1], padded->nb[2],
                                ggml_row_size(padded->type, start))));

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1],
                        k * model.layers[il].ple_conv1d->nb[0]));
        // this kernel keeps the file type, so cast it before it multiplies an f32 activation
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, shifted, wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, gated, conv_out));
}
