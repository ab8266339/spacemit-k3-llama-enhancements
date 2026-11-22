# Q6_K decode experiments

The stock standalone trace put a large fraction of single-token decode samples inside a Q6_K matrix operation, RVV copies and synchronization. We tested two alternatives for the existing single-row IME2 path. These patches are experiment inputs. The default launcher uses the vendor backend.

- `direct.patch` keeps the quantized activation in scratch and streams repacked Q6_K weights directly to the dot kernel in groups of four column tiles.
- `grouped.patch` keeps weight staging and the existing paired-worker synchronization. It groups up to four adjacent tiles when the scratch capacity and the number of output columns permit it.

| Local GCC 15.2 build | 128-token prefill, tokens/s | 64-token decode, tokens/s | Output-head GEMV, ms |
|---|---:|---:|---:|
| Unchanged control | 80.49 | 8.82 | 18.23 |
| Direct weights | 80.37 | 8.64 | 20.14 |
| Grouped TCM copies | 80.81 | 8.68 | 19.11 |
| Unchanged control, repeat | 81.34 | 8.81 | — |

The last column is K=2048, N=248320 and one input row. Each operator measurement used one warm-up and 11 timed calls. All FP32 output bytes matched the control on six shapes, including N=1056 to exercise the final partial group of tiles. The 12 short next-token probe records also matched byte for byte between all four locally built full-model runs. Both candidates reduced median decode throughput on this GGUF, so we kept the vendor kernel.

## Reproduce the operator test

Use the release's headers and libraries together. `RUNTIME` points to the extracted llama.cpp release. `SPINE` points to the extracted spine-runtime release. Set `LD_LIBRARY_PATH` for the vendor's runtime dependencies as on your normal K3 installation.

```bash
c++ -O2 -std=c++17 tools/q6_gemv.cpp -I"$RUNTIME/include" \
  -L"$RUNTIME/lib" -Wl,-rpath-link,"$RUNTIME/lib" -Wl,-rpath-link,"$SPINE/lib" \
  -lggml -lggml-base -o q6_gemv
export GGML_SPACEMIT_BACKEND=1 GGML_SPACEMIT_WORKERS=8
export SPACEMIT_PERFER_CORE_ARCH=a064 SPACEMIT_PERFER_CORE_ID=8,9,10,11,12,13,14,15
./q6_gemv 2048 248320 11 output.f32
```

If your binary archive omits the ggml headers, add `-I"$LLAMA_SRC/ggml/include"` from the same tagged source. The probe initializes deterministic weights and activations, quantizes every weight row, runs the standalone device, and writes the full FP32 output. Repeat for each backend library and compare files with `cmp`. Finite outputs and matching bytes test agreement with the same-version control, not agreement with an unquantized model.

## Rebuild a candidate

Check out [SpacemiT llama.cpp v0.2.5](https://github.com/spacemit-com/llama.cpp/tree/v0.2.5). Use a clean checkout for each variant. Apply one patch with `git -C "$LLAMA_SRC" apply /absolute/path/to/direct.patch` or `grouped.patch`. For the control leave the checkout clean.

```bash
cmake -S experiments/q6-decode -B build-q6 -G Ninja \
  -DLLAMA_SRC="$LLAMA_SRC" -DSPERT_DIR="$SPINE" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DGGML_SPACEMIT=ON -DGGML_NATIVE=OFF \
  -DGGML_CPU_RISCV64_SPACEMIT=OFF -DGGML_OPENMP=OFF \
  -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
cmake --build build-q6 --target ggml-spacemit --parallel 8
```

Use `LD_PRELOAD=/absolute/path/to/the/built/libggml-spacemit.so.0` for the tested process with the release runtime libraries on its loader path. Compare locally built baseline and candidate libraries from the same compiler, leaving system libraries in place. [Raw rounds, hashes and all six operator shapes](../../results/q6-experiments.json).
