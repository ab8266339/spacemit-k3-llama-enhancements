// Deterministic Q6_K GEMV probe for the SpacemiT standalone GGML backend.
// Build against the headers and libraries from the same llama.cpp release.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static size_t positive(const char * text, size_t limit) {
    char * end = nullptr;
    const unsigned long long n = std::strtoull(text, &end, 10);
    if (*text == '\0' || *text == '-' || *end != '\0' || n == 0 || n > limit) {
        throw std::invalid_argument("invalid dimension or repeat count");
    }
    return static_cast<size_t>(n);
}

static float next_value(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return static_cast<float>(state >> 8) * 0x1p-23f - 1.0f;
}

static int run(int argc, char ** argv) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: %s K N REPEATS OUTPUT.f32\n", argv[0]);
        return 2;
    }
    const size_t k = positive(argv[1], 32768);
    const size_t n = positive(argv[2], 1048576);
    const size_t repeats = positive(argv[3], 1000);
    if (k % 256 || n % 32 || k * n > (size_t{1} << 30)) {
        throw std::invalid_argument("require K divisible by 256, N divisible by 32, at most 2^30 weights");
    }
    ggml_backend_load_all();
    ggml_backend_t backend = ggml_backend_init_by_name("SPACEMIT0", nullptr);
    if (!backend) throw std::runtime_error("SPACEMIT0 unavailable; check GGML_SPACEMIT_BACKEND and your runtime");
    ggml_context * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!ctx) throw std::runtime_error("GGML context allocation failed");
    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, k, n);
    ggml_set_name(weights, "output.weight");
    ggml_tensor * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, k);
    ggml_tensor * output = ggml_mul_mat(ctx, weights, input);
    if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("backend rejected this GEMV");
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_backend_buffer_t storage = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!storage) throw std::runtime_error("tensor allocation failed");

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q6_K, k);
    std::vector<uint8_t> encoded(row_bytes * n);
    std::vector<float> row(k);
    uint32_t state = 179;
    for (size_t i = 0; i < n; ++i) {
        for (float & x : row) x = next_value(state);
        const size_t written = ggml_quantize_chunk(GGML_TYPE_Q6_K, row.data(), encoded.data() + i * row_bytes, 0, 1, k, nullptr);
        if (written != row_bytes) throw std::runtime_error("quantizer returned an unexpected byte count");
    }
    ggml_backend_tensor_set(weights, encoded.data(), 0, encoded.size());
    std::vector<uint8_t>().swap(encoded);
    for (float & x : row) x = next_value(state);
    ggml_backend_tensor_set(input, row.data(), 0, ggml_nbytes(input));

    std::vector<double> ms;
    for (size_t rep = 0; rep <= repeats; ++rep) {
        const auto start = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("graph failed");
        ggml_backend_synchronize(backend);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        if (rep) ms.push_back(std::chrono::duration<double, std::milli>(elapsed).count());
    }
    std::vector<float> values(n);
    ggml_backend_tensor_get(output, values.data(), 0, ggml_nbytes(output));
    if (!std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }) ||
        std::all_of(values.begin(), values.end(), [](float x) { return x == 0.0f; })) {
        throw std::runtime_error("invalid output");
    }
    std::ofstream file(argv[4], std::ios::binary);
    file.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
    file.close();
    if (!file) throw std::runtime_error("could not write output");
    std::sort(ms.begin(), ms.end());
    const double median = (ms[(repeats - 1) / 2] + ms[repeats / 2]) / 2;
    std::printf("{\"k\":%zu,\"n\":%zu,\"repeats\":%zu,\"median_ms\":%.6f}\n", k, n, repeats, median);
    ggml_backend_buffer_free(storage);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}

int main(int argc, char ** argv) {
    try { return run(argc, argv); }
    catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
