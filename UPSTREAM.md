# Upstream projects

Snapshot of GitHub repository metadata on 2026-11-22. Subscribers are GitHub's `subscribers_count` (watching notifications); stars are `stargazers_count`.

| Repository | Stars | Subscribers | Relevance |
|---|---:|---:|---|
| [SpacemiT llama.cpp](https://github.com/spacemit-com/llama.cpp) | 16 | 1 | Maintains the K3 standalone device, IME2 kernels and vendor release tested here. |
| [SpacemiT docs-ai](https://github.com/spacemit-com/docs-ai) | 7 | 0 | User-facing bilingual deployment guide. [PR #174](https://github.com/spacemit-com/docs-ai/pull/174) adds device discovery and a tested server command. |
| [SpacemiT spine-runtime](https://github.com/spacemit-com/spine-runtime) | 4 | 0 | The scheduling and per-worker scratch-buffer ABI used by the device. |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | 136,058 | 907 | Broad upstream for reusable CPU/RVV changes. [#31897](https://github.com/ggml-org/llama.cpp/pull/31897) already routes Gated DeltaNet to the SpacemiT RVV kernel. |
| [ggml-org/ggml](https://github.com/ggml-org/ggml) | 16,157 | 151 | Tensor library. Check the corresponding llama.cpp backend and upstream synchronization before proposing a general change. |
| [Ollama](https://github.com/ollama/ollama) | 187,842 | 1,040 | Relevant to runtime packaging and application integration. The K3 vendor kernel experiments in this repo are scoped to llama.cpp. |
| [platima/llama.cpp-SpacemiT-K3](https://github.com/platima/llama.cpp-SpacemiT-K3) | 0 | 0 | Community K3 fork with board-specific work. |

The vendor already implements [the standalone device](https://github.com/spacemit-com/llama.cpp/pull/43), [MoE/dense Q4 acceleration](https://github.com/spacemit-com/llama.cpp/pull/53), and [RVV/TCM recurrent GDN](https://github.com/spacemit-com/llama.cpp/pull/54). [Op-list caching #82](https://github.com/spacemit-com/llama.cpp/pull/82) and [split-K FP16 work #79](https://github.com/spacemit-com/llama.cpp/pull/79) are related efforts. The two Q6_K experiments did not pass the full-model speed gate and have not been proposed as upstream performance patches.
