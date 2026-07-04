#include <iostream>
#include <iomanip>
#include <cstdlib>
#include "hypute_ml.h"

int main(int argc, char* argv[]) {
    const char* out_path = (argc > 1) ? argv[1] : "model.hml";

    HyputeMLConfig cfg{};
    cfg.vocab_size    = 50000;
    cfg.embedding_dim = 256;
    cfg.hidden_dim    = 1024;
    cfg.num_layers    = 12;
    cfg.output_dim    = 50000;

    std::cout << "[HAYAKO ML] Initialising model...\n";
    HyputeMLModel* model = hypute_ml_create(&cfg);
    if (!model) { std::cerr << "[ERROR] Model creation failed.\n"; return 1; }

    hypute_ml_randomize_weights(model, 2026);

    size_t params = hypute_ml_param_count(model);
    size_t mem    = hypute_ml_memory_bytes(model);
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[HAYAKO ML] Parameters : " << params / 1e6 << " M\n";
    std::cout << "[HAYAKO ML] Memory     : " << mem / (1024.0 * 1024.0) << " MB\n";

    std::cout << "[HAYAKO ML] Saving weights to: " << out_path << "\n";
    int rc = hypute_ml_save_weights(model, out_path);
    if (rc != 0) { std::cerr << "[ERROR] Save failed: " << rc << "\n"; return 1; }

    std::cout << "[HAYAKO ML] Done.\n";
    hypute_ml_destroy(model);
    return 0;
}
