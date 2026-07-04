#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <random>
#include "hypute_ml_lm.h"

// ── Vocab I/O ─────────────────────────────────────────────────────────────────
// Minimal JSON parser for a flat {"word": id, ...} object.

static std::unordered_map<std::string, uint32_t> load_vocab(const char* path) {
    std::unordered_map<std::string, uint32_t> v;
    std::ifstream f(path);
    if (!f) { std::cerr << "[ERROR] Cannot open " << path << "\n"; return v; }
    std::string line, key;
    uint32_t val;
    // Simple line-by-line scan: find "word": number
    std::string full((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    size_t pos = 0;
    while ((pos = full.find('"', pos)) != std::string::npos) {
        ++pos;
        size_t end = full.find('"', pos);
        if (end == std::string::npos) break;
        key = full.substr(pos, end - pos);
        pos = end + 1;
        size_t colon = full.find(':', pos);
        if (colon == std::string::npos) break;
        pos = colon + 1;
        while (pos < full.size() && (full[pos] == ' ' || full[pos] == '\n')) ++pos;
        val = static_cast<uint32_t>(std::stoul(full.substr(pos)));
        v[key] = val;
    }
    return v;
}

// ── Sampling ──────────────────────────────────────────────────────────────────

static uint32_t sample_token(const float* logits, size_t vocab_size,
                              float temperature, std::mt19937& rng) {
    if (temperature <= 0.0f) {
        // Greedy
        return static_cast<uint32_t>(
            std::max_element(logits, logits + vocab_size) - logits);
    }
    // Softmax with temperature
    std::vector<float> probs(vocab_size);
    float max_l = *std::max_element(logits, logits + vocab_size);
    float sum   = 0.0f;
    for (size_t i = 0; i < vocab_size; ++i) {
        probs[i] = std::exp((logits[i] - max_l) / temperature);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    std::discrete_distribution<uint32_t> dist(probs.begin(), probs.end());
    return dist(rng);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: lm_generate <lm.hgru> <lm_vocab.json> "
                     "[prompt] [gen_len] [temperature]\n";
        return 1;
    }

    const char* model_path = argv[1];
    const char* vocab_path = argv[2];
    std::string prompt      = (argc > 3) ? argv[3] : "the";
    int         gen_len     = (argc > 4) ? std::atoi(argv[4]) : 100;
    float       temperature = (argc > 5) ? std::atof(argv[5]) : 0.8f;

    // ── Load vocab ────────────────────────────────────────────────────────────
    auto word2id = load_vocab(vocab_path);
    if (word2id.empty()) return 1;
    std::vector<std::string> id2word(word2id.size());
    for (const auto& kv : word2id) {
        if (kv.second < id2word.size()) id2word[kv.second] = kv.first;
    }

    auto tokenise = [&](const std::string& text) {
        std::vector<uint32_t> ids;
        std::istringstream ss(text);
        std::string w;
        while (ss >> w) {
            std::transform(w.begin(), w.end(), w.begin(), ::tolower);
            auto it = word2id.find(w);
            ids.push_back(it != word2id.end() ? it->second : 1u);  // 1 = <unk>
        }
        return ids;
    };

    // ── Load model ────────────────────────────────────────────────────────────
    HyputeLMConfig cfg{};
    cfg.vocab_size = 10000;
    cfg.embed_dim  = 256;
    cfg.hidden_dim = 512;
    cfg.num_layers = 2;

    HyputeLMModel* model = hypute_lm_create(&cfg);
    if (!model) { std::cerr << "[ERROR] Model creation failed.\n"; return 1; }

    int rc = hypute_lm_load_weights(model, model_path);
    if (rc != 0) { std::cerr << "[ERROR] load_weights: " << rc << "\n"; return 1; }

    std::cout << "\n  Model     : " << hypute_lm_param_count(model) / 1e6 << " M params\n";
    std::cout << "  Memory    : " << hypute_lm_memory_bytes(model) / (1024.0 * 1024.0)
              << " MB  (int8)\n";
    std::cout << "  Prompt    : \"" << prompt << "\"\n";
    std::cout << "  Tokens    : " << gen_len << "\n";
    std::cout << "  Temperature: " << temperature << "\n\n";

    // ── Prime hidden state with prompt ────────────────────────────────────────
    hypute_lm_reset_state(model);
    std::vector<float>    logits(cfg.vocab_size);
    std::vector<uint32_t> prompt_ids = tokenise(prompt);

    for (uint32_t id : prompt_ids)
        hypute_lm_step(model, id, logits.data(), logits.size());

    // ── Generate ──────────────────────────────────────────────────────────────
    std::mt19937 rng(42);
    std::cout << "  " << prompt;

    uint32_t last_id = prompt_ids.empty() ? 0u : prompt_ids.back();
    double total_ns = 0.0;

    for (int i = 0; i < gen_len; ++i) {
        hypute_lm_step(model, last_id, logits.data(), logits.size());
        total_ns += hypute_lm_last_step_ns(model);

        last_id = sample_token(logits.data(), cfg.vocab_size, temperature, rng);
        const std::string& word = (last_id < id2word.size()) ? id2word[last_id] : "<unk>";
        std::cout << " " << word;
        std::cout.flush();
    }

    double avg_us = total_ns / gen_len / 1e3;
    std::cout << "\n\n  Avg step: " << std::fixed << std::setprecision(1)
              << avg_us << " µs/token  ("
              << 1e6 / avg_us << " tokens/sec)\n\n";

    hypute_lm_destroy(model);
    return 0;
}
