# Upstream sync ledger

How this tree follows [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), what it keeps as its own, and the
checklist for the next merge. [HALO-HYBRID.md](HALO-HYBRID.md) describes the changes themselves.

## Base

| | |
|---|---|
| Last merge | upstream `ec7630a64` (b11344, 2026-10-01), merged in four stages: `2a829def2`, `cd9ed2a1d`, `e1f188251`, `d8bc069d6` |
| Fork delta | `git diff ec7630a64 -- <file>` is the authoritative fork change to a file; `halo-hybrid` comments only help reading it |
| Taken since, ahead of the next merge | `134b2bb75` (CPY to a transposed non-contiguous dst), `51ce9c11a` (per-thread padding memset), `8ae386707` (ggml_acc out of bounds), `65840ed53` (CLAMP non-contiguous), a hand port of `dd266785c` / `fc9ce6b9d` (MMQ src1 padding) |

## Method

Merges, never rebases: `main` is published. A sync is a series of staged merges of upstream commits in a separate
worktree, each stage built and tested before the next, with its conflict resolutions written in the merge message
and summarised in a row below. Upstream pull requests are not opened from this tree.

| Sync | Stages | Notes |
|---|---|---|
| 2026-10-01, to `ec7630a64` | 4 | dual graph arenas apply to lane 0; this tree's MMQ kept (upstream's W4A4 `prec_src1` is Blackwell-only); upstream GLM-5.3 support (`glm5-next`) carried beside this tree's `glm5next`; the MTP driver ported to `common_batch`; upstream's qwen4exp k-pool / MTP rewrite (#29751, #29761) not taken (same design as this tree's) |

## Ownership

| Kind | Files | At a merge |
|---|---|---|
| Ours | `src/models/qwen4exp.*`, `conversion/qwen4exp.py`, `src/llama-memory-hybrid-idx.*`, `src/llama-kv-cache-kpool.*`, `src/models/glm5next.*` and its converter and vision code, `ggml/src/ggml-cuda/mmq*.cu/.cuh` and their instances, `fattn-mma-f16.cuh` + `fattn-swizzle.cuh`, `mmvq.cu`, `ggml/src/ggml-rpc/*` | keep ours; read upstream's commits to these files one by one and port, mark not applicable, or defer (list them in the sync's row) |
| Union | `src/llama-arch.*`, hparams, `gguf-py` tables, op enums (`ggml.h`, `ggml.c`), backend iface structs | take both sides; the fork's entries stay in their `halo-hybrid` blocks |
| Theirs | everything else | resolve from upstream, then re-apply the fork's hunks |

## Declined or carried differently

| Upstream | Here |
|---|---|
| #24364 Blackwell FP4 activation precision (W4A4/W4A8, `GGML_CUDA_MMQ_PREC`) | not taken; this tree's MMQ is RDNA-tuned |
| #28536, #29152, #28770, #29298 NVIDIA flash-attention changes | not taken; this tree keeps its `fattn-mma-f16.cuh` and `fattn-swizzle.cuh` |
| #29440 RDMA completion channel | not taken (RDMA is off on the test hosts; the polling loop stays) |
| #29751, #29761 qwen4exp k-pool and MTP rewrite | not taken; carried as this tree's qwen4exp |
| `glm5-next` converter route | upstream's architecture is carried and loads; this tree's converter writes `glm5next` |
| #29184 shared-expert fusion | covered here by the MoE-tail kernel and quantize-once |
| #30103 | fixed differently (`fattn-mma-f16.cuh`) |
| #27694 probabilistic draft sampling (after the base) | this tree has its own rejection sampling; decide at the next merge |
| RPC protocol | fork-local major 107 and command ids from 128; upstream's RPC changes are ported, never merged as-is |

## Checklist for the next merge

1. `git diff <old> <new> -- ggml/include/ggml.h ggml/src/ggml.c ggml/include/ggml-rpc.h ggml/src/ggml-backend-impl.h`:
   op enum and count, iface order, upstream's backend API version (bump `GGML_BACKEND_API_VERSION` = upstream's + 1000).
2. `git log -S'LLM_ARCH_GLM5_NEXT' -S'LLM_ARCH_QWEN4EXP' <old>..<new> -- src common tools`: mirror new sites of the twin
   architectures.
3. Backend registry names in the strided-mask allowlist, converter routes in `conversion/__init__.py`.
4. `.github`: `git rm` every modify/delete conflict and read newly added upstream workflows before keeping any.
5. Lane-aware scopes (`prepare_ubatch`, `extract_outputs`, `graph_reserve`, `graph_get_cb`) must not regain
   `sched.get()`.
6. Speculative decoding: `batch.set_embd` in the early ingest becomes `set_embd_state` when upstream 781dbc5ac lands;
   trim the draft queue next to `result.resize(dp.n_max)`.
7. `generate_cu_files.py` must still emit the three DKQ=512 sparse instances (regenerate: zero diff).
8. Per-file triage of upstream commits to the "ours" files (above).
9. Wire changes: rebuild both RPC hosts; bump the fork's protocol minor (or major) on any wire change.
10. Update `UPSTREAM_BASE` in `tests/halo/lib/env_check.py`, the prebuilt UI version in `tools/ui/CMakeLists.txt`
    (`b` + upstream's `git rev-list --count <base>`) and the base row above; run `tests/halo` base full and the
    GPU suites before the merge lands on `main`.

## Markers

In upstream's files a fork hunk carries a `// halo-hybrid: why` comment, and fork entries in upstream's enums and
tables sit between `// halo-hybrid ...: begin` and `// halo-hybrid ...: end` lines (the op enum and its name tables,
the scheduler declarations in `ggml-backend.h`). Files only this tree has need no markers. Markers help reading; the
diff against the base is what counts.

## Adding a fork op

A new `GGML_OP_*`: add it at the end of the enum's `halo-hybrid ops` block and of the two name tables in `ggml.c`,
give it a CPU implementation (the reference for test-backend-ops), a `supports_op` answer on every backend that can
see it (RPC included: the server answers), test-backend-ops cases, and bump the RPC protocol (the handshake compares
`GGML_OP_COUNT`).

## Version strings

`git describe --tags --match 'halo-*'` is the version; tags are dated (`halo-YYYY-MM-DD`) and pushed by name.
`build-info.cmake` is upstream's (build number = commit count) until a merge decides otherwise.
