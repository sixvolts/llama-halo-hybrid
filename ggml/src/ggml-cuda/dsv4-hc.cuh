#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// halo-hybrid: DSV4_HC_MIX with the preceding DSV4_HC_POST (hc_post, whose dst is the prologue's x; may be null) and
// the following RMS_NORM -> MUL(w) of the pre-mix row (rms_norm/mul; may be null) in the same launch. The MUL's dst
// also gets a registered q8_1 copy at <= 8 tokens. write_out=false skips the un-normed row (only the norm reads it).
void ggml_cuda_op_dsv4_hc_mix_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * hc_post, const ggml_tensor * rms_norm, ggml_tensor * mul, bool write_out);
