"""
Train a GRU word-level language model on WikiText-2, then export to .hgru.

Architecture: Embedding(VOCAB, EMBED) → GRU(HIDDEN, NUM_LAYERS) → Linear(VOCAB)

Run: python train_lm.py
Outputs: lm.hgru       (int8 weights for the C++ engine)
         lm_fp32.pt    (full fp32 checkpoint for PyTorch comparison)
         lm_vocab.json (word→id mapping)
"""

import re, json, struct, os, math, time
import numpy as np
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("pip install torch")

try:
    from datasets import load_dataset
except ImportError:
    raise SystemExit("pip install datasets")

# ── Hyper-parameters ──────────────────────────────────────────────────────────
VOCAB_SIZE  = 10000
EMBED_DIM   = 256
HIDDEN_DIM  = 512
NUM_LAYERS  = 2
EPOCHS      = 20
SEQ_LEN     = 64   # BPTT truncation length
BATCH_SIZE  = 32
LR          = 0.002
CLIP        = 0.5  # gradient clipping
DROPOUT     = 0.3

# ── Tokeniser ─────────────────────────────────────────────────────────────────

def build_vocab(text, n=VOCAB_SIZE):
    words = re.findall(r'\S+', text.lower())
    counts = Counter(words)
    vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
    for w, _ in counts.most_common(n - 3):
        vocab[w] = len(vocab)
    return vocab

def encode(text, vocab):
    return [vocab.get(w, 1) for w in re.findall(r'\S+', text.lower())]

# ── Model ─────────────────────────────────────────────────────────────────────

class GRULanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding   = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.drop        = nn.Dropout(DROPOUT)
        self.gru_cells   = nn.ModuleList()
        for i in range(NUM_LAYERS):
            in_d = EMBED_DIM if i == 0 else HIDDEN_DIM
            self.gru_cells.append(nn.GRUCell(in_d, HIDDEN_DIM))
        self.output_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
        # Tie embedding and output weights for better perplexity
        if EMBED_DIM == HIDDEN_DIM:
            self.output_proj.weight = self.embedding.weight

    def forward(self, token_ids, h_states):
        """
        token_ids: (batch,)
        h_states: list of (batch, hidden) tensors, one per layer
        returns: logits (batch, vocab), new_h_states
        """
        x = self.drop(self.embedding(token_ids))
        new_h = []
        for i, cell in enumerate(self.gru_cells):
            x = cell(x, h_states[i])
            new_h.append(x)
            x = self.drop(x)
        return self.output_proj(x), new_h

    def init_hidden(self, batch_size, device):
        return [torch.zeros(batch_size, HIDDEN_DIM, device=device)
                for _ in self.gru_cells]


# ── Training helpers ──────────────────────────────────────────────────────────

def batchify(data, batch_size):
    """Reshape flat token list into (num_batches, batch_size)."""
    n = (len(data) // batch_size) * batch_size
    t = torch.tensor(data[:n], dtype=torch.long)
    return t.view(batch_size, -1).t().contiguous()  # (total_len/B, B)

def get_batch(source, i, seq_len=SEQ_LEN):
    seq_len = min(seq_len, source.size(0) - 1 - i)
    data   = source[i : i + seq_len]
    target = source[i + 1 : i + 1 + seq_len].reshape(-1)
    return data, target

def detach(h):
    return [x.detach() for x in h]

def evaluate(model, data_source, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    h = model.init_hidden(BATCH_SIZE, device)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(data_source, i)
            data, targets = data.to(device), targets.to(device)
            h = detach(h)
            for t in range(data.size(0)):
                logits, h = model(data[t], h)
            # only measure loss on last step for speed
            total_loss += criterion(logits, targets[-BATCH_SIZE:]).item() * BATCH_SIZE
            n += BATCH_SIZE
    return total_loss / n


# ── .hgru export ──────────────────────────────────────────────────────────────

HGRU_MAGIC   = 0x48475255
HGRU_VERSION = 1

def quantise_matrix(w_np):
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

def save_hgru(path, model):
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', HGRU_MAGIC, HGRU_VERSION))
        f.write(struct.pack('<4Q', VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS))

        write_qmatrix(f, model.embedding.weight.detach().cpu().numpy())

        for i, cell in enumerate(model.gru_cells):
            # PyTorch GRUCell: weight_ih (3H × in_d), rows = [r; z; n]
            in_d = EMBED_DIM if i == 0 else HIDDEN_DIM
            wih = cell.weight_ih.detach().cpu().numpy()  # (3H, in_d)
            whh = cell.weight_hh.detach().cpu().numpy()  # (3H, H)
            H   = HIDDEN_DIM
            write_qmatrix(f, wih[:H])     # W_r
            write_qmatrix(f, wih[H:2*H])  # W_z
            write_qmatrix(f, wih[2*H:])   # W_n
            write_qmatrix(f, whh[:H])     # U_r
            write_qmatrix(f, whh[H:2*H])  # U_z
            write_qmatrix(f, whh[2*H:])   # U_n

        write_qmatrix(f, model.output_proj.weight.detach().cpu().numpy())

    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {path}  ({size_mb:.1f} MB  int8)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load WikiText-2
    print("Loading WikiText-2 …")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_text = "\n".join(ds['train']['text'])
    val_text   = "\n".join(ds['validation']['text'])
    test_text  = "\n".join(ds['test']['text'])

    print("Building vocabulary …")
    vocab = build_vocab(train_text, VOCAB_SIZE)
    id2word = {v: k for k, v in vocab.items()}
    with open('lm_vocab.json', 'w') as f:
        json.dump(vocab, f)
    print(f"  Vocab size: {len(vocab)}")

    train_ids = encode(train_text, vocab)
    val_ids   = encode(val_text,   vocab)
    test_ids  = encode(test_text,  vocab)
    print(f"  Train tokens: {len(train_ids):,}  Val: {len(val_ids):,}  Test: {len(test_ids):,}")

    train_data = batchify(train_ids, BATCH_SIZE).to(device)
    val_data   = batchify(val_ids,   BATCH_SIZE).to(device)
    test_data  = batchify(test_ids,  BATCH_SIZE).to(device)

    model     = GRULanguageModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    params = sum(p.numel() for p in model.parameters())
    print(f"\n  Parameters: {params/1e6:.1f} M  ({params*4/1e6:.0f} MB fp32)\n")

    best_val_loss = float('inf')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, steps = 0.0, 0
        h = model.init_hidden(BATCH_SIZE, device)

        for i in range(0, train_data.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(train_data, i)
            data, targets = data.to(device), targets.to(device)
            h = detach(h)
            optimizer.zero_grad()

            loss = torch.tensor(0.0, device=device)
            for t in range(data.size(0)):
                logits, h = model(data[t], h)
                loss = loss + criterion(logits, targets[t * BATCH_SIZE : (t+1) * BATCH_SIZE])
            loss = loss / data.size(0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        train_ppl = math.exp(total_loss / steps)
        val_loss  = evaluate(model, val_data, criterion, device)
        val_ppl   = math.exp(val_loss)
        scheduler.step(val_loss)

        marker = " *" if val_loss < best_val_loss else ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'lm_best.pt')

        print(f"  Epoch {epoch:2d}/{EPOCHS}  "
              f"train_ppl {train_ppl:6.1f}  val_ppl {val_ppl:6.1f}{marker}")

    # Load best checkpoint
    model.load_state_dict(torch.load('lm_best.pt', map_location='cpu'))
    model = model.cpu()

    test_loss = evaluate(model, test_data.cpu(), criterion, 'cpu')
    test_ppl  = math.exp(test_loss)
    print(f"\n  Test perplexity: {test_ppl:.1f}")

    fp32_mb  = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    print(f"\nExporting …")
    save_hgru('lm.hgru', model)
    torch.save(model.state_dict(), 'lm_fp32.pt')
    print(f"  Saved lm_fp32.pt  ({fp32_mb:.1f} MB  fp32)")
    print(f"  Saved lm_vocab.json")
    print(f"\nSize: {fp32_mb:.0f} MB  fp32  →  {os.path.getsize('lm.hgru')/1e6:.1f} MB  int8")
    print("Done. Run compare_lm.py next.")

if __name__ == '__main__':
    main()
