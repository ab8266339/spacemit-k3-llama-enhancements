# SpacemiT K3 llama.cpp enhancements

Run Qwen3.6-35B-A3B on the eight A100 cores of a SpacemiT K3 and compare the vendor's legacy CPU path with its standalone `SPACEMIT0` device. The launcher and benchmark use Python 3's standard library.

在 K3 真机上，选择独立后端并将物理批次调到 128，使这份 Q4_0 模型的输入处理从 34.62 提升到 83.26 token/s，生成从 7.59 提升到 8.89 token/s。

## Run on a K3

The tested system used Bianbu 4.0.4, the Linux riscv64 [SpacemiT llama.cpp v0.2.5 release](https://github.com/spacemit-com/llama.cpp/releases/tag/v0.2.5), [Spine runtime 0.7.2](https://github.com/spacemit-com/spine-runtime/releases/tag/0.7.2) and [SpacemiT ONNX Runtime 2.0.8](https://github.com/spacemit-com/onnxruntime/releases/tag/2.0.8). The vendor binaries need their shared libraries even for a text-only run. If using unpacked archives, pass their library directories with repeated `--lib-dir` options. A system installation can use the dynamic loader's usual paths.

```bash
MODEL=/path/to/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf
SERVER=/path/to/spacemit-llama.cpp/bin/llama-server
python3 k3_llama.py serve --server "$SERVER" --model "$MODEL"
```

The service listens on `127.0.0.1:8080`, using one slot, a 1024-token context, eight threads, and a physical/logical batch of 128. The startup log should name `SPACEMIT0`. Ctrl-C stops it. The older path can be selected with `--profile legacy64`. These controls are for the **SpacemiT A100 RISC-V cores**.

## Reproduce the comparison

```bash
sha256sum "$MODEL"
python3 k3_llama.py bench --server "$SERVER" --model "$MODEL" --out runs
```

The benchmark starts and stops each server in sequence. It verifies the server identity, the number of tokens actually processed, absence of prompt-cache hits, and identical input IDs across profiles. Each unique run directory retains logs, prompts' token hashes, generated text, per-round timings and medians. To test one configuration, add `--profiles standalone128`.

| Configuration | Batch / microbatch | Prompt tokens/s | Generated tokens/s |
|---|---:|---:|---:|
| Vendor CPU path on eight A100 cores | 64 / 64 | 34.62 | 7.59 |
| Vendor standalone device | 64 / 64 | 64.93 | 8.82 |
| Vendor standalone device | 128 / 128 | 83.26 | 8.89 |

These are medians of three measured rounds after one warm-up, with 128 uncached input tokens and 64 greedy output tokens. At matched batch sizes, the device change raised input throughput by 87.5% and generation by 16.2%. Increasing the batch size addresses a separate limit. The speedups above come from selecting settings already present in the vendor runtime. The model is [bartowski's Q4_0 GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/blob/main/Qwen_Qwen3.6-35B-A3B-Q4_0.gguf), SHA-256 `52312daa5b2190c1f5723d33c3315c01c55af4206f6c6e6eb63f3d8dd52bb85e`.

## What the chip was doing

- Linux sampled the test process on CPU IDs 8 through 15. On the legacy path, a separate diagnostic recorded that each worker's 384 KiB scratch buffer lies entirely inside a real `/dev/tcm` mapping. The platform TCM monitor nevertheless reported eight free blocks during these runs.
- Both paths hit IME2 matrix kernels. In the separate legacy prefill trace, 582 of 662 samples with `ggml_vec_dot_f32` as their leaf had the generic Gated DeltaNet routine as a caller. The standalone trace hit the vendor RVV Gated DeltaNet implementation.
- In the standalone decode trace, roughly 23% of user-mode A100 samples fell in synchronization functions and 18% in RVV `memcpy1d`. These fractions describe program-counter samples, not matrix-unit occupancy or DRAM bandwidth.
- The two production backends answered both short answer checks correctly. Their first-choice next token matched on 11 of 12 short prefixes, and longer greedy continuations differed. The vendor documents [lossy router conversion](https://github.com/spacemit-com/llama.cpp/pull/53). The measurements establish throughput; model quality needs a workload-specific evaluation.

See [machine-readable results and sampling counts](results/qwen36-35b-k3.json). The kernel implementation is maintained in [SpacemiT's llama.cpp fork](https://github.com/spacemit-com/llama.cpp). The [main llama.cpp project](https://github.com/ggml-org/llama.cpp) also maintains RISC-V CPU support. Codex assisted with tooling and board tests; Grok Build assisted with the source audit.
