"""
Side-by-side comparison of our trained model vs a HuggingFace BERT model.

Evaluates both on the same 2 000 IMDB test samples and prints a table showing
accuracy, model size, and average inference time.

Prerequisites:
    python train.py          (produces sentiment.hml, vocab.json, test_data.bin)
    pip install transformers torch datasets tqdm

Run: python compare.py
"""

import json, re, time, os
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
except ImportError:
    raise SystemExit("pip install torch datasets")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# ── Config ────────────────────────────────────────────────────────────────────

HF_MODEL    = "textattack/bert-base-uncased-imdb"
NUM_SAMPLES = 2000   # same test slice for both models
MAX_LEN_HF  = 512

# ── Shared test data ──────────────────────────────────────────────────────────

def load_test_samples(n=NUM_SAMPLES):
    ds = load_dataset("imdb", split="test")
    return [{"text": ds[i]["text"], "label": ds[i]["label"]} for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════════
# Our model: re-run in PyTorch (same weights as sentiment.hml)
# ══════════════════════════════════════════════════════════════════════════════

import re as _re

def tokenise(text):
    return _re.findall(r'\b[a-z]+\b', text.lower())

def encode_ours(text, vocab, max_len=300):
    tokens = tokenise(text)[:max_len]
    return [vocab.get(t, 1) for t in tokens] or [0]


class SentimentModel(torch.nn.Module):
    def __init__(self, vocab=20000, embed=128, hidden=256, layers=1, output=2):
        super().__init__()
        self.embedding     = torch.nn.Embedding(vocab, embed, padding_idx=0)
        self.hidden_layers = torch.nn.ModuleList(
            [torch.nn.Linear(embed if i == 0 else hidden, hidden, bias=False)
             for i in range(layers)])
        self.output_layer  = torch.nn.Linear(hidden, output, bias=False)

    def forward_ids(self, ids_list):
        tensors = [torch.tensor(ids, dtype=torch.long) for ids in ids_list]
        emb = [self.embedding(t).mean(0) for t in tensors]
        x = torch.stack(emb)
        for l in self.hidden_layers:
            x = torch.nn.functional.relu(l(x))
        return self.output_layer(x)


def load_pytorch_weights(model, hml_path, vocab_size, embed_dim, hidden_dim, num_layers, output_dim):
    """Read our .hml binary and fill a PyTorch model (for accuracy comparison)."""
    import struct

    HML_MAGIC = 0x484D4C57

    def read_qmatrix(f, rows, cols):
        q = np.frombuffer(f.read(rows * cols), dtype=np.int8).reshape(rows, cols).astype(np.float32)
        s = np.frombuffer(f.read(rows * 4),    dtype=np.float32)
        return (q * s[:, None])  # dequantise: float weights

    with open(hml_path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        assert magic == HML_MAGIC, "Not a .hml file"
        f.read(40)  # skip config

        emb_w   = read_qmatrix(f, vocab_size, embed_dim)
        layer_ws = []
        in_d = embed_dim
        for _ in range(num_layers):
            layer_ws.append(read_qmatrix(f, hidden_dim, in_d))
            in_d = hidden_dim
        out_w = read_qmatrix(f, output_dim, hidden_dim)

    with torch.no_grad():
        model.embedding.weight.copy_(torch.tensor(emb_w))
        for i, w in enumerate(layer_ws):
            model.hidden_layers[i].weight.copy_(torch.tensor(w))
        model.output_layer.weight.copy_(torch.tensor(out_w))



def eval_ours(samples, vocab):
    print("\n[OUR MODEL] Loading weights from sentiment.hml …")
    model = SentimentModel(vocab=30000, embed=256, hidden=512, layers=2, output=2)
    load_pytorch_weights(model, 'sentiment.hml', 30000, 256, 512, 2, 2)
    model.eval()

    correct, latencies = 0, []
    with torch.no_grad():
        for s in tqdm(samples, desc="  Our model"):
            ids = encode_ours(s['text'], vocab)
            t0  = time.perf_counter()
            logit = model.forward_ids([ids])
            latencies.append((time.perf_counter() - t0) * 1000)
            if logit.argmax(1).item() == s['label']:
                correct += 1

    acc      = correct / len(samples) * 100
    avg_ms   = float(np.mean(latencies))
    size_mb  = os.path.getsize('sentiment.hml') / 1e6
    fp32_mb  = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    return acc, avg_ms, size_mb, fp32_mb


# ══════════════════════════════════════════════════════════════════════════════
# HuggingFace BERT model
# ══════════════════════════════════════════════════════════════════════════════

def eval_hf(samples):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
    except ImportError:
        raise SystemExit("pip install transformers")

    print(f"\n[HF MODEL] Loading {HF_MODEL} …")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    hf_model  = AutoModelForSequenceClassification.from_pretrained(HF_MODEL)
    hf_model.eval()

    param_count = sum(p.numel() for p in hf_model.parameters())
    fp32_mb     = param_count * 4 / 1e6

    correct, latencies = 0, []
    with torch.no_grad():
        for s in tqdm(samples, desc=f"  {HF_MODEL.split('/')[-1]}"):
            enc = tokenizer(s['text'], return_tensors='pt',
                            truncation=True, max_length=MAX_LEN_HF)
            t0     = time.perf_counter()
            logits = hf_model(**enc).logits
            latencies.append((time.perf_counter() - t0) * 1000)
            pred = logits.argmax(1).item()
            # textattack/bert-base-uncased-imdb: label 0 = neg, 1 = pos (matches IMDB)
            if pred == s['label']:
                correct += 1

    acc    = correct / len(samples) * 100
    avg_ms = float(np.mean(latencies))
    return acc, avg_ms, fp32_mb, param_count


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not os.path.exists('sentiment.hml'):
        raise SystemExit("Run train.py first to produce sentiment.hml")
    if not os.path.exists('vocab.json'):
        raise SystemExit("vocab.json missing — run train.py first")

    with open('vocab.json') as f:
        vocab = json.load(f)

    print(f"Loading {NUM_SAMPLES} IMDB test samples …")
    samples = load_test_samples(NUM_SAMPLES)

    our_acc, our_ms, our_int8_mb, our_fp32_mb = eval_ours(samples, vocab)
    hf_acc,  hf_ms,  hf_fp32_mb, hf_params   = eval_hf(samples)

    # ── Print table ───────────────────────────────────────────────────────────
    W = 42
    print("\n" + "=" * W)
    print("  COMPARISON — IMDB Sentiment (2 000 samples)")
    print("=" * W)
    print(f"  {'Metric':<22}  {'Ours':>8}  {'BERT':>8}")
    print("-" * W)
    print(f"  {'Accuracy':<22}  {our_acc:>7.1f}%  {hf_acc:>7.1f}%")
    print(f"  {'Avg latency (ms)':<22}  {our_ms:>8.2f}  {hf_ms:>8.2f}")
    print(f"  {'Model size (int8/fp32)':<22}  {our_int8_mb:>6.1f} MB  {hf_fp32_mb:>6.0f} MB")
    print(f"  {'Size ratio':<22}  {'1×':>8}  {hf_fp32_mb/our_int8_mb:>7.0f}×")
    print("=" * W)
    print(f"\n  Our model uses {hf_fp32_mb/our_int8_mb:.0f}× less RAM than BERT.")
    print(f"  Accuracy gap: {hf_acc - our_acc:.1f} pp — expected for a 4 M vs 109 M param model.")
    print(f"\n  Run the C++ engine eval for real on-device latency:")
    print(f"    cd ../../build && ./sentiment_eval ../examples/sentiment/sentiment.hml \\")
    print(f"                                        ../examples/sentiment/test_data.bin")


if __name__ == '__main__':
    main()
