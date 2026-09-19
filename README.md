# llama.cpp for Strix Halo, with and without a GPU next to it

I've had a Strix Halo board for about a year and been playing around with it for various projects when it's not
just being a beefy linux machine. I also grabbed an R9700 Pro AI card late last year for another machine, thinking 
it would be fun to compare the two. I ended up parting out the machine the R9700 was in for something else and 
wondered what might be possible with the R9700 in the Strix Halo machine. On the Framework desktop board, there's 
an x4 4.0 slot hanging out. I already had an x4 extension cable so I could mount a 25G card in it, but a GPU 
would fit just fine too. I have my board in a Fractal Design case instead of the framework shell (bought the bare 
board), so I had plenty of room for the card and my power supply had the new 12V connector. Even with today's 
pricing, a Framework Strix Halo 128GB board and an R9700 is about ~5k all in, so similar price to a DGX spark 
but with a little more RAM (~160GB, obv with caveats), and it's a regular 16-core ryzen PC instead of the tacky gold box.

![The build: Framework Strix Halo board with the R9700 on an x4 riser, Noctua on the APU, Seasonic PSU](docs/halo-hybrid/build.jpeg)

So, the kicker is that it works. The model this tree is built around is **Qwen3.8-Flash-Next** (unsloth
UD-Q4_K_XL, 111 GB): on the Strix Halo plus the R9700 it decodes at **45 tok/s** (52 greedy) with the model's own
MTP draft head and prefills at **~1,500 tok/s**, where stock llama.cpp on the same layout does 27-28 tok/s. The
same tree also runs the 200 GB GLM-5.3-Flash across two of these boxes over a 100G link. (The setup was put
together on Qwen3.5-122B, 24 → 49 tok/s; 3.8 came out the week it was working, and the numbers below are 3.8's.)

Here's how it works. We can't just slap part of the model on the R9700 and expect it to be good though. It's
actually worse if you try to do that in most cases. First, we need to place the parts of the model that benefit
from the different parts of the hardware. So, with a big MoE model like this, we have a bunch of data that only
gets touched for some tokens and those routed experts need to get put on the Strix in the bigger unified memory
pool. Most of the model is experts, but we might only read a couple of GB of it per token. The dense parts of the
model get touched for every token, so we put them on the R9700 where we have more compute and memory bandwidth:
KV cache, the dense trunk, and critically, the MTP drafter. We can stuff the remaining VRAM on the R9700 with as
many whole expert layers as fit. This all works because only a few KB of data per token needs to cross that narrow
x4 4.0 link, so as long as the latency isn't bad, it doesn't matter. Trying to do something like Tensor Parallelism
across these two would not work well because of that bottleneck.

The launch line for that layout:

```
LLAMA_PREFILL_LANES=2 \
llama-server -m Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  -dev ROCm0,ROCm1 -ts 1,0 --fit off -fa on -ngl 999 -c 8192 -b 4096 -ub 1024 --load-mode none -np 1 \
  -ot 'blk\.(1[4-9]|[2-4][0-9])\.ffn_(gate|up|down)_exps=ROCm1,per_layer_token_embd=CPU' \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf -devd ROCm0 -ngld 999 \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --host 0.0.0.0 --port 8080
```

I kept going on tuning, and tried to reduce the number of kernel launches, which seemed to be holding back
performance. I wasn't hitting anywhere near the right numbers per the theoretical bandwidth for each device. The
kernel and scheduler changes that came out of that are listed in [HALO-HYBRID.md](HALO-HYBRID.md), and every one
of them is measured on the iGPU alone as well, because the tree has to stay useful on a Strix Halo with no card.

## Which configuration?

The same tree runs five hardware shapes. Each row links to the launch line, the memory budget and the numbers in
the [cookbook](docs/halo-hybrid/COOKBOOK.md); the kernel and scheduler changes behind them are in
[HALO-HYBRID.md](HALO-HYBRID.md).

| Hardware | Model it runs here | Status | Numbers |
|---|---|---|---|
| [One Strix Halo, nothing else](docs/halo-hybrid/COOKBOOK.md#1-one-strix-halo-by-itself) | anything up to ~115 GB | runs | the baseline the hybrids are measured against (24 tok/s on Qwen3.5-122B) |
| [One Strix Halo + one R9700](docs/halo-hybrid/COOKBOOK.md#2-one-strix-halo--one-r9700) | Qwen3.8-Flash-Next, Qwen3.5-122B | production | 45 tok/s (52 greedy), ~1,500 tok/s prefill |
| [Two Strix Halos over RDMA](docs/halo-hybrid/COOKBOOK.md#3-two-strix-halos-over-rdma) | GLM-5.3-Flash (200 GB) | derived, not measured | |
| [Two Strix Halos + one R9700 on the head node](docs/halo-hybrid/COOKBOOK.md#4-two-strix-halos--one-r9700-on-the-head-node) | GLM-5.3-Flash | production | 517 tok/s prefill / 20.5 tok/s decode at 13K, 503 / 20.7 at 26K |
| [Two Strix Halos + one R9700 on each](docs/halo-hybrid/COOKBOOK.md#5-two-strix-halos--one-r9700-on-each) | GLM-5.3-Flash | planned, card ordered | estimate 23-25 tok/s |

## Qwen3.8-Flash-Next: one box, with or without the card

The hybrid-N table for different context budgets, the per-stream numbers with and without the draft head, and the
iGPU-only baseline are in the cookbook: [recipe 1](docs/halo-hybrid/COOKBOOK.md#1-one-strix-halo-by-itself) (no
card) and [recipe 2](docs/halo-hybrid/COOKBOOK.md#2-one-strix-halo--one-r9700) (with the R9700). The draft head is
the `shared-Q8_0` file in the unsloth repo's `MTP/` folder; it borrows the target's embeddings and lm head.

## GLM-5.3-Flash across two Strix Halo boxes

The 200 GB GLM-5.3-Flash (unsloth UD-Q4_K_XL) runs split between two 128 GB Strix Halo boxes over a direct 100G
link (Intel E810) with llama.cpp's RPC backend over RDMA, with an R9700 on the head node: KV and the MTP draft
head on the R9700, layers 0-24 on the head node (dense trunk on the card, experts on the iGPU), layers 25-44 on
the second box's unified memory. Single stream, 128K context: 20.5 tok/s decode at 13K with the model's own MTP
head (14 without it), 517 tok/s prefill. It took two scheduler fixes, an RDMA transport fix and a loader fix,
all on `main` (upstream's GLM-5.3-Flash PR is not merged yet). Launch lines, the rpc-server unit and the
operating rules: [cookbook, recipe 4](docs/halo-hybrid/COOKBOOK.md#4-two-strix-halos--one-r9700-on-the-head-node);
the draft-head export and the fixes: [HALO-HYBRID.md](HALO-HYBRID.md) ("GLM-5.3-Flash across two hosts").

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
