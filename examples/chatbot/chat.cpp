#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <random>
#include "hypute_ml_lm.h"

// ── Minimal JSON vocab loader ─────────────────────────────────────────────────

static void load_vocab(const char* path,
                       std::unordered_map<std::string,uint32_t>& w2id,
                       std::vector<std::string>& id2w) {
    std::ifstream f(path);
    if (!f) { std::cerr << "[ERROR] Cannot open vocab: " << path << "\n"; return; }
    std::string all((std::istreambuf_iterator<char>(f)), {});
    size_t pos = 0;
    while ((pos = all.find('"', pos)) != std::string::npos) {
        size_t ks = ++pos, ke = all.find('"', pos); if (ke==std::string::npos) break;
        std::string key = all.substr(ks, ke-ks); pos = ke+1;
        size_t col = all.find(':', pos); if (col==std::string::npos) break; pos=col+1;
        while (pos<all.size()&&(all[pos]==' '||all[pos]=='\n')) ++pos;
        uint32_t id = static_cast<uint32_t>(std::stoul(all.substr(pos)));
        w2id[key] = id;
    }
    id2w.resize(w2id.size());
    for (auto& kv : w2id)
        if (kv.second < id2w.size()) id2w[kv.second] = kv.first;
}

// ── Tokenise a line (mirrors train_chatbot.py's words() function) ─────────────

static std::vector<uint32_t> tokenise(const std::string& text,
                                       const std::unordered_map<std::string,uint32_t>& w2id) {
    std::vector<uint32_t> ids;
    std::string cur;
    for (char c : text) {
        if (std::isalpha(c) || c == '\'' ) {
            cur += std::tolower(c);
        } else {
            if (!cur.empty()) {
                auto it = w2id.find(cur);
                ids.push_back(it != w2id.end() ? it->second : 1u);
                cur.clear();
            }
            // punctuation as own tokens
            std::string p(1, c);
            if (p=="."||p==","||p=="!"||p=="?"||p=="-") {
                auto it = w2id.find(p);
                if (it != w2id.end()) ids.push_back(it->second);
            }
        }
    }
    if (!cur.empty()) {
        auto it = w2id.find(cur);
        ids.push_back(it != w2id.end() ? it->second : 1u);
    }
    return ids;
}

// ── Top-k sampling ────────────────────────────────────────────────────────────

static uint32_t sample_topk(const float* logits, size_t vocab_size,
                              float temperature, int top_k, std::mt19937& rng) {
    // Collect top-k indices
    std::vector<std::pair<float,uint32_t>> scores(vocab_size);
    for (size_t i = 0; i < vocab_size; ++i) scores[i] = {logits[i], (uint32_t)i};
    int k = std::min(top_k, (int)vocab_size);
    std::partial_sort(scores.begin(), scores.begin()+k, scores.end(),
                      [](auto& a, auto& b){ return a.first > b.first; });

    // Softmax over top-k
    std::vector<float> probs(k);
    float mx = scores[0].first, sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        probs[i] = std::exp((scores[i].first - mx) / temperature);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;

    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    return scores[dist(rng)].second;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: chat <chatbot.hgru> <chatbot_vocab.json> "
                     "[temperature=0.75] [top_k=40] [max_tokens=60]\n";
        return 1;
    }

    float temperature = (argc > 3) ? std::atof(argv[3]) : 0.75f;
    int   top_k       = (argc > 4) ? std::atoi(argv[4]) : 40;
    int   max_tokens  = (argc > 5) ? std::atoi(argv[5]) : 60;

    // ── Vocab ─────────────────────────────────────────────────────────────────
    std::unordered_map<std::string,uint32_t> w2id;
    std::vector<std::string> id2w;
    load_vocab(argv[2], w2id, id2w);
    if (w2id.empty()) return 1;

    uint32_t USER_ID = w2id.count("<user>") ? w2id["<user>"] : 3;
    uint32_t SYS_ID  = w2id.count("<sys>")  ? w2id["<sys>"]  : 4;
    uint32_t EOS_ID  = w2id.count("<eos>")  ? w2id["<eos>"]  : 2;

    // ── Model ─────────────────────────────────────────────────────────────────
    HyputeLMConfig cfg{};
    cfg.vocab_size = 16000;
    cfg.embed_dim  = 512;
    cfg.hidden_dim = 1024;
    cfg.num_layers = 3;

    HyputeLMModel* model = hypute_lm_create(&cfg);
    if (!model) { std::cerr << "[ERROR] create failed\n"; return 1; }
    if (hypute_lm_load_weights(model, argv[1]) != 0) {
        std::cerr << "[ERROR] load_weights failed\n"; return 1;
    }

    std::mt19937 rng(std::random_device{}());
    std::vector<float> logits(cfg.vocab_size);

    std::cout << "\n========================================\n";
    std::cout << "  HYPUTE ML CHATBOT\n";
    std::cout << "  " << hypute_lm_param_count(model)/1e6 << " M params  |  "
              << hypute_lm_memory_bytes(model)/(1024.0*1024.0) << " MB (int8)\n";
    std::cout << "  Type a message, press Enter. Ctrl-C to quit.\n";
    std::cout << "========================================\n\n";

    hypute_lm_reset_state(model);
    std::string line;

    while (true) {
        std::cout << "You: ";
        if (!std::getline(std::cin, line) || line.empty()) break;

        // Feed  <user> [tokens] <sys>  to prime the hidden state
        hypute_lm_step(model, USER_ID, logits.data(), logits.size());
        for (uint32_t id : tokenise(line, w2id))
            hypute_lm_step(model, id, logits.data(), logits.size());
        hypute_lm_step(model, SYS_ID, logits.data(), logits.size());

        // Generate response until <user>, <eos>, or max_tokens
        std::cout << "Bot: ";
        std::string response;
        for (int i = 0; i < max_tokens; ++i) {
            uint32_t next = sample_topk(logits.data(), cfg.vocab_size,
                                        temperature, top_k, rng);
            if (next == USER_ID || next == EOS_ID) break;

            const std::string& w = (next < id2w.size()) ? id2w[next] : "";
            if (w.empty() || w[0]=='<') continue;  // skip unknowns / special

            // Space before word, no space before punctuation
            bool is_punct = (w=="."||w==","||w=="!"||w=="?"||w=="-");
            if (!response.empty() && !is_punct) response += ' ';
            response += w;

            hypute_lm_step(model, next, logits.data(), logits.size());
        }

        // Capitalise first letter
        if (!response.empty()) response[0] = std::toupper(response[0]);
        if (!response.empty() && response.back() != '.' &&
            response.back() != '!' && response.back() != '?')
            response += '.';

        std::cout << response << "\n\n";
    }

    hypute_lm_destroy(model);
    return 0;
}
