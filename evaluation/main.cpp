#include <iostream>
#include <iomanip>
#include "hypute_ml.h"

int main() {
    HyputeMLConfig cfg{};
    cfg.vocab_size    = 1000;
    cfg.embedding_dim = 64;
    cfg.hidden_dim    = 128;
    cfg.num_layers    = 2;
    cfg.output_dim    = 100;

    HyputeMLModel* model = hypute_ml_create(&cfg);
    if (!model) { std::cerr << "[ERROR] Model creation failed.\n"; return 1; }

    hypute_ml_randomize_weights(model, 42);

    uint64_t inputs[] = {1, 7, 42};
    float    scores[100];
    int      n = hypute_ml_infer(model, inputs, 3, scores, 100);
    if (n <= 0) { std::cerr << "[ERROR] Inference failed.\n"; return 1; }

    std::cout << "========================================================\n";
    std::cout << "            HYPUTE ML ENVIRONMENT VERIFICATION          \n";
    std::cout << "========================================================\n\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Parameters    : " << hypute_ml_param_count(model) << "\n";
    std::cout << "  Memory        : " << hypute_ml_memory_bytes(model) / 1024.0 << " KB\n";
    std::cout << "  Latency       : " << hypute_ml_last_latency_ns(model) << " ns\n";
    std::cout << "  Outputs       : " << n << " scores\n";
    std::cout << "  First score   : " << scores[0] << "\n\n";
    std::cout << "========================================================\n";

    hypute_ml_destroy(model);
    return 0;
}
