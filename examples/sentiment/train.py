"""
Train a bag-of-words sentiment classifier on IMDB, then export to .hml.

Architecture (maps 1:1 onto HyputeMLConfig):
  Embedding(VOCAB, EMBED)  →  mean-pool
  →  Linear(EMBED, HIDDEN) → ReLU
  →  Linear(HIDDEN, HIDDEN) → ReLU   (×  NUM_LAYERS - 1)
  →  Linear(HIDDEN, 2)

Run: python train.py
Outputs: sentiment.hml  (weights for the C++ engine)
         vocab.json     (token → id mapping for the C++ evaluator)
         test_data.bin  (pre-tokenised test samples for C++ eval)
"""

import re, json, struct, time
import numpy as np
from collections import Counter

# ── Optional tqdm ────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

# ── Hyper-parameters ─────────────────────────────────────────────────────────
VOCAB_SIZE  = 30000
EMBED_DIM   = 256
HIDDEN_DIM  = 512
NUM_LAYERS  = 2       # maps to HyputeMLConfig.num_layers
OUTPUT_DIM  = 2
EPOCHS      = 8
LR          = 0.0005
BATCH_SIZE  = 64
MAX_LEN     = 400     # tokens per review

# ── Imports that might not be present ────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    raise SystemExit("pip install torch")

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("pip install datasets")


# ── Tokeniser ────────────────────────────────────────────────────────────────

def tokenise(text):
    return re.findall(r'\b[a-z]+\b', text.lower())


def build_vocab(texts):
    counter = Counter()
    for t in tqdm(texts, desc="Building vocab"):
        counter.update(tokenise(t))
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in counter.most_common(VOCAB_SIZE - 2):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab, max_len=MAX_LEN):
    tokens = tokenise(text)[:max_len]
    return [vocab.get(t, 1) for t in tokens] or [0]


# ── Dataset ───────────────────────────────────────────────────────────────────

class IMDBDataset(Dataset):
    def __init__(self, examples, vocab):
        self.data = [(encode(ex['text'], vocab), ex['label']) for ex in examples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, label = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), label


def collate(batch):
    ids_list, labels = zip(*batch)
    lengths = [len(x) for x in ids_list]
    max_l   = max(lengths)
    padded  = torch.zeros(len(ids_list), max_l, dtype=torch.long)
    for i, ids in enumerate(ids_list):
        padded[i, :len(ids)] = ids
    return padded, torch.tensor(labels, dtype=torch.long), torch.tensor(lengths, dtype=torch.long)


# ── Model ─────────────────────────────────────────────────────────────────────

class SentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding    = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(EMBED_DIM, HIDDEN_DIM, bias=False))
        for _ in range(NUM_LAYERS - 1):
            self.hidden_layers.append(nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=False))
        self.output_layer = nn.Linear(HIDDEN_DIM, OUTPUT_DIM, bias=False)

    def forward(self, ids, lengths):
        # Mean-pool embeddings (ignore padding)
        emb = self.embedding(ids)                           # (B, T, E)
        mask = (ids != 0).float().unsqueeze(-1)             # (B, T, 1)
        x = (emb * mask).sum(1) / mask.sum(1).clamp(min=1) # (B, E)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


# ── .hml export ──────────────────────────────────────────────────────────────

HML_MAGIC   = 0x484D4C57
HML_VERSION = 1

def quantise_matrix(w_np):
    """(rows, cols) float32 → int8 quantised + per-row float scales."""
    rows, cols = w_np.shape
    q = np.zeros((rows, cols), dtype=np.int8)
    s = np.zeros(rows, dtype=np.float32)
    for i in range(rows):
        mx = float(np.abs(w_np[i]).max())
        if mx == 0.0:
            s[i] = 1.0
        else:
            s[i] = mx / 127.0
            q[i] = np.clip(np.round(w_np[i] / s[i]), -127, 127).astype(np.int8)
    return q, s


def write_qmatrix(f, w_np):
    q, s = quantise_matrix(w_np.astype(np.float32))
    f.write(q.tobytes())
    f.write(s.tobytes())


def save_hml(path, model):
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', HML_MAGIC, HML_VERSION))
        f.write(struct.pack('<5Q',
            VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM))

        write_qmatrix(f, model.embedding.weight.detach().numpy())
        for layer in model.hidden_layers:
            write_qmatrix(f, layer.weight.detach().numpy())
        write_qmatrix(f, model.output_layer.weight.detach().numpy())

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {path}  ({size_mb:.2f} MB)")


# ── Test-data binary for C++ evaluator ───────────────────────────────────────

def save_test_bin(path, examples, vocab, n=2000):
    """Write N test samples as a compact binary for the C++ evaluator."""
    samples = examples[:n]
    with open(path, 'wb') as f:
        f.write(struct.pack('<Q', len(samples)))
        for ex in samples:
            ids = encode(ex['text'], vocab)
            f.write(struct.pack('<B', ex['label']))
            f.write(struct.pack('<I', len(ids)))
            f.write(struct.pack(f'<{len(ids)}Q', *ids))
    print(f"  Saved {path}  ({len(samples)} samples)")


# ── Main ──────────────────────────────────────────────────────────────────────

import os

def main():
    print("Loading IMDB dataset …")
    ds = load_dataset("imdb")

    print("Building vocabulary …")
    vocab = build_vocab([ex['text'] for ex in ds['train']])
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    print(f"  Vocab size: {len(vocab)}")

    train_ds = IMDBDataset(ds['train'], vocab)
    test_ds  = IMDBDataset(ds['test'],  vocab)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate)
    test_loader  = DataLoader(test_ds,  BATCH_SIZE, shuffle=False, collate_fn=collate)

    model     = SentimentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}  ({params*4/1e6:.1f} MB fp32)")

    print("\nTraining …\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, correct, n = 0.0, 0, 0
        for ids, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            optimizer.zero_grad()
            logits = model(ids, lengths)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            n          += len(labels)
        train_acc = correct / n * 100

        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for ids, labels, lengths in test_loader:
                correct += (model(ids, lengths).argmax(1) == labels).sum().item()
                n       += len(labels)
        val_acc = correct / n * 100
        print(f"  Epoch {epoch}: train {train_acc:.1f}%  val {val_acc:.1f}%  loss {total_loss/n:.4f}")

    print("\nExporting …")
    save_hml('sentiment.hml', model)
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    print("  Saved vocab.json")
    save_test_bin('test_data.bin', list(ds['test']), vocab)

    fp32_mb = params * 4 / 1e6
    int8_mb = os.path.getsize('sentiment.hml') / 1e6
    print(f"\nModel size: {fp32_mb:.1f} MB  (fp32)  →  {int8_mb:.1f} MB  (int8, {fp32_mb/int8_mb:.1f}× smaller)")
    print("Done. Run compare.py next.")


if __name__ == '__main__':
    main()
