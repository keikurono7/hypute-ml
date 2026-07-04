#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include "hypute_ml.h"

// Read test_data.bin written by train.py:
//   uint64  num_samples
//   for each sample:
//     uint8   label
//     uint32  num_tokens
//     uint64[num_tokens]  token_ids
struct Sample {
    uint8_t              label;
    std::vector<uint64_t> ids;
};

static std::vector<Sample> load_test_bin(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "[ERROR] Cannot open " << path << "\n"; return {}; }

    uint64_t n;
    f.read(reinterpret_cast<char*>(&n), 8);
    std::vector<Sample> out;
    out.reserve(n);

    for (uint64_t i = 0; i < n; ++i) {
        Sample s;
        uint8_t  label;
        uint32_t ntok;
        f.read(reinterpret_cast<char*>(&label), 1);
        f.read(reinterpret_cast<char*>(&ntok),  4);
        s.label = label;
        s.ids.resize(ntok);
        f.read(reinterpret_cast<char*>(s.ids.data()),
               static_cast<std::streamsize>(ntok * 8));
        if (!f) break;
        out.push_back(std::move(s));
    }
    return out;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: sentiment_eval <model.hml> <test_data.bin>\n";
        return 1;
    }

    // ── Load model ────────────────────────────────────────────────────────────
    HyputeMLConfig cfg{};
    cfg.vocab_size    = 30000;
    cfg.embedding_dim = 256;
    cfg.hidden_dim    = 512;
    cfg.num_layers    = 2;
    cfg.output_dim    = 2;

    HyputeMLModel* model = hypute_ml_create(&cfg);
    if (!model) { std::cerr << "[ERROR] Model creation failed.\n"; return 1; }

    int rc = hypute_ml_load_weights(model, argv[1]);
    if (rc != 0) { std::cerr << "[ERROR] load_weights returned " << rc << "\n"; return 1; }

    hypute_ml_set_dense_mode(model, 1);

    // ── Load test data ────────────────────────────────────────────────────────
    auto samples = load_test_bin(argv[2]);
    if (samples.empty()) { std::cerr << "[ERROR] No samples loaded.\n"; return 1; }

    // ── Evaluate ──────────────────────────────────────────────────────────────
    std::vector<float>  scores(cfg.output_dim);
    std::vector<double> latencies;
    latencies.reserve(samples.size());

    size_t correct = 0;
    for (const auto& s : samples) {
        hypute_ml_infer(model, s.ids.data(), s.ids.size(),
                        scores.data(), scores.size());
        latencies.push_back(hypute_ml_last_latency_ns(model));

        int pred = (scores[1] > scores[0]) ? 1 : 0;
        if (pred == static_cast<int>(s.label)) ++correct;
    }

    std::sort(latencies.begin(), latencies.end());
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double p50 = latencies[latencies.size() / 2];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];
    double acc  = 100.0 * correct / samples.size();

    // ── Print results ──────────────────────────────────────────────────────────
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n========================================\n";
    std::cout << "  HYPUTE ML — C++ ENGINE EVALUATION\n";
    std::cout << "========================================\n\n";
    std::cout << "  Dataset       : IMDB sentiment\n";
    std::cout << "  Samples       : " << samples.size() << "\n";
    std::cout << "  Parameters    : " << hypute_ml_param_count(model) / 1e6 << " M\n";
    std::cout << "  Memory        : " << hypute_ml_memory_bytes(model) / (1024.0 * 1024.0) << " MB\n\n";
    std::cout << "  Accuracy      : " << acc << " %\n";
    std::cout << "  Avg latency   : " << avg / 1e3 << " µs\n";
    std::cout << "  P50 latency   : " << p50 / 1e3 << " µs\n";
    std::cout << "  P99 latency   : " << p99 / 1e3 << " µs\n";
    std::cout << "  Throughput    : " << 1e9 / avg / 1000.0 << " K inferences/sec\n";
    std::cout << "\n========================================\n";

    hypute_ml_destroy(model);
    return 0;
}
