# Contributing to llama-halo-hybrid

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for AMD Strix Halo, alone or with
one Radeon card or a second machine over RPC. The root [CONTRIBUTING.md](../CONTRIBUTING.md) and
[AGENTS.md](../AGENTS.md) are upstream's; this file covers the fork.

- **Bug reports:** use the issue forms. Setup, card, ROCm version, kernel command line, build line, launch line and
  `LLAMA_*` / `GGML_*` environment variables are what a hybrid setup needs for triage.
- **Changes that are not specific to this fork** (general llama.cpp features and fixes) belong upstream; this fork
  picks them up when it syncs.
- **Pull requests:** open an issue first for anything beyond a small fix, so the approach can be agreed. Changes on
  the default path need to keep working on the APU alone, the APU with one dGPU, and two machines over RPC, and
  must not change numerics without saying so.
- **AI use:** you are responsible for every line you submit, however it was produced. Say in the PR how AI was
  used, and write the description yourself.
