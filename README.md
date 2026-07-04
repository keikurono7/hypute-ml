# Hypute ML

**Run large neural networks in under 10 MB of RAM. No GPU. No BLAS. No runtime dependencies.**

Hypute ML is a C++17 inference engine built for low-end devices — phones, microcontrollers, embedded Linux — where standard runtimes don't fit. The core uses a proprietary architecture shipped as a static library; the implementation is not exposed in this repository.

---

## vs DialoGPT-small — same task, real hardware

Both models ran the same 10-turn conversation on a laptop CPU (no GPU):

| | **Hypute ML** | DialoGPT-small |
|---|---|---|
| Parameters | 8.5 M | 124 M |
| **Model size** | **9 MB** | 498 MB |
| **Avg latency** | **154 ms** | 8 311 ms |
| Size ratio | **1×** | 55× larger |
| Speed ratio | **1×** | 54× slower |
| Fits in 64 MB RAM | ✅ | ❌ |

DialoGPT-small requires ~500 MB at runtime. Our model runs the same task in 9 MB — **55× smaller, 54× faster.**

---

## 75 M-parameter benchmark

> vocab 50 000 · embed 256 · hidden 1024 · 12 layers · single CPU core

| Metric | Value |
|---|---|
| Model memory | **72.9 MB** |
| Avg latency | **83.9 ms** / inference |
| P99 latency | 90.1 ms |
| Throughput | ~12 inferences / sec |
| Compute efficiency | **75%** |

Full CI results: [GitHub Actions → benchmark workflow](.github/workflows/benchmark.yml)

---

## Why it fits where others don't

**Compact weight storage.** Weights use a proprietary compressed format — the 75M benchmark fits in 73 MB vs ~300 MB for a standard fp32 model.

**Proprietary sparse compute.** The engine skips work that doesn't affect the output. On the 75M benchmark, ~75% of compute is eliminated per inference — with no accuracy loss.

**No runtime deps.** Pure C++17, auto-vectorised by gcc. No BLAS, no ONNX, no TFLite, no Python. Link one `.a` file and ship.

---

## What you can build

| Task | Example | Model size |
|---|---|---|
| Text classification | IMDB sentiment | ~17 MB |
| Language model | WikiText-2 text generation | ~9 MB |
| Chatbot | Cornell Movie Dialogs dialogue | **9 MB** |
| Custom | Any vocab/hidden/layer config | scales linearly |

---

## Quick start

```bash
git clone https://github.com/keikurono7/hypute-ml
cd hypute-ml
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Run the benchmark

```bash
# 75M-param model, 500 runs
HML_VOCAB=50000 HML_EMBED=256 HML_HIDDEN=1024 \
HML_LAYERS=12 HML_OUTPUT=50000 HML_RUNS=500 \
./hypute_ml_bench
```

### Run the chatbot

```bash
# requires chatbot.hgru + chatbot_vocab.json (see examples/chatbot/train_chatbot.py)
./chat chatbot.hgru chatbot_vocab.json
```

---

## C API

```c
#include "hypute_ml.h"

// Create a 75M-param model
HyputeMLConfig cfg = {
    .vocab_size    = 50000,
    .embedding_dim = 256,
    .hidden_dim    = 1024,
    .num_layers    = 12,
    .output_dim    = 50000,
};
HyputeMLModel* m = hypute_ml_create(&cfg);
hypute_ml_load_weights(m, "weights.hml");

// Run inference
uint64_t ids[] = {1, 7, 42};
float    scores[50000];
hypute_ml_infer(m, ids, 3, scores, 50000);

// Read telemetry
printf("latency:    %.2f ms\n",  hypute_ml_last_latency_ns(m) / 1e6);
printf("memory:     %zu MB\n",   hypute_ml_memory_bytes(m) / 1000000);
printf("efficiency: %.1f%%\n",
    100.0 * (1.0 - (double)hypute_ml_activation_count(m)
                         / hypute_ml_compute_slots(m)));

hypute_ml_destroy(m);
```

### Language model / chatbot API

```c
#include "hypute_ml_lm.h"

HyputeLMConfig cfg = { .vocab_size=8000, .embed_dim=256, .hidden_dim=512, .num_layers=2 };
HyputeLMModel* m = hypute_lm_create(&cfg);
hypute_lm_load(m, "chatbot.hgru");

float logits[8000];
hypute_lm_reset_state(m);
hypute_lm_step(m, token_id, logits, 8000);  // stateful — call per token

hypute_lm_destroy(m);
```

Full declarations: [`hypute_ml.h`](hypute_ml.h) · [`hypute_ml_lm.h`](hypute_ml_lm.h)

---

## Repo layout

```
hypute-ml/
├── hypute_ml.h              — classification engine API
├── hypute_ml_lm.h           — language model / chatbot API
├── libhypute_ml.a           — pre-compiled classification engine
├── libhypute_ml_lm.a        — pre-compiled LM engine
├── CMakeLists.txt
├── benchmarks/
│   └── inference/
│       └── benchmark.cpp    — configurable latency benchmark
├── evaluation/
│   └── main.cpp             — smoke test
├── tools/
│   └── generate_weights.cpp — generate random .hml weights
└── examples/
    ├── sentiment/
    │   ├── train.py         — train IMDB sentiment classifier (PyTorch → .hml)
    │   └── eval.cpp         — C++ evaluator, reports accuracy + latency
    ├── lm/
    │   ├── train_lm.py      — train WikiText-2 language model (PyTorch → .hgru)
    │   └── generate.cpp     — top-k temperature text generation
    └── chatbot/
        ├── train_chatbot.py — train on Cornell Movie Dialogs (→ .hgru)
        ├── chat.cpp         — interactive C++ chat loop
        └── compare_transformer.py — side-by-side vs DialoGPT-small
```

---

## Requirements

- `g++` ≥ 9 (C++17)
- `cmake` ≥ 3.12
- `libpthread` (standard on Linux)
- No other runtime dependencies

---

## Licence

Source code is proprietary — the engine is distributed as a compiled static library. Header files and example code in this repository are MIT licensed.
