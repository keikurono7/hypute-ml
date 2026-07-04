#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "hypute_ml.h"

static size_t env_or(const char* name, size_t def) {
    const char* v = std::getenv(name);
    return v ? static_cast<size_t>(std::stoull(v)) : def;
}

int main() {
    HyputeMLConfig cfg{};
    cfg.vocab_size    = env_or("HML_VOCAB",    50000);
    cfg.embedding_dim = env_or("HML_EMBED",    256);
    cfg.hidden_dim    = env_or("HML_HIDDEN",   1024);
    cfg.num_layers    = env_or("HML_LAYERS",   12);
    cfg.output_dim    = env_or("HML_OUTPUT",   50000);

    size_t runs       = env_or("HML_RUNS",     10000);
    size_t seq_len    = env_or("HML_SEQ",      8);

    std::cout << "========================================================\n";
    std::cout << "        HYPUTE ML INFERENCE BENCHMARK\n";
    std::cout << "========================================================\n\n";
    std::cout << "  Config\n";
    std::cout << "    vocab_size    : " << cfg.vocab_size    << "\n";
    std::cout << "    embedding_dim : " << cfg.embedding_dim << "\n";
    std::cout << "    hidden_dim    : " << cfg.hidden_dim    << "\n";
    std::cout << "    num_layers    : " << cfg.num_layers    << "\n";
    std::cout << "    output_dim    : " << cfg.output_dim    << "\n";
    std::cout << "    sequence_len  : " << seq_len           << "\n";
    std::cout << "    runs          : " << runs              << "\n\n";

    HyputeMLModel* model = hypute_ml_create(&cfg);
    if (!model) { std::cerr << "[ERROR] Model creation failed.\n"; return 1; }

    hypute_ml_randomize_weights(model, 1337);

    size_t params  = hypute_ml_param_count(model);
    size_t mem     = hypute_ml_memory_bytes(model);
    std::cout << "  Model\n";
    std::cout << "    Parameters    : " << params / 1000000.0 << " M\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "    Memory        : " << mem / (1024.0 * 1024.0) << " MB\n\n";

    // Warmup
    std::vector<uint64_t> ids(seq_len);
    std::vector<float>    scores(cfg.output_dim);
    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < seq_len; ++j) ids[j] = (i * seq_len + j) % cfg.vocab_size;
        hypute_ml_infer(model, ids.data(), seq_len, scores.data(), cfg.output_dim);
    }

    // Benchmark
    std::vector<double> latencies(runs);
    size_t total_active = 0, total_slots = 0;

    for (size_t r = 0; r < runs; ++r) {
        for (size_t j = 0; j < seq_len; ++j) ids[j] = (r * seq_len + j) % cfg.vocab_size;
        hypute_ml_infer(model, ids.data(), seq_len, scores.data(), cfg.output_dim);
        latencies[r]  = hypute_ml_last_latency_ns(model);
        total_active += hypute_ml_activation_count(model);
        total_slots  += hypute_ml_compute_slots(model);
    }

    std::sort(latencies.begin(), latencies.end());
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / runs;
    double p50 = latencies[runs / 2];
    double p99 = latencies[static_cast<size_t>(runs * 0.99)];
    double throughput = 1e9 / avg;
    double efficiency = total_slots > 0
        ? 100.0 * (1.0 - static_cast<double>(total_active) / total_slots) : 0.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Results\n";
    std::cout << "    Avg Latency      : " << avg        << " ns\n";
    std::cout << "    P50 Latency      : " << p50        << " ns\n";
    std::cout << "    P99 Latency      : " << p99        << " ns\n";
    std::cout << "    Throughput       : " << throughput / 1000.0 << " K inferences/sec\n";
    std::cout << "    Compute Efficiency: " << efficiency  << " %\n\n";
    std::cout << "========================================================\n";

    hypute_ml_destroy(model);
    return 0;
}
