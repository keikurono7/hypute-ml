# ─────────────────────────────────────────────────────────────────────────────
# Hypute ML — GRU Chatbot Training (Colab / Kaggle)
#
# Run all cells top to bottom.
# At the end:
#   Colab  → auto-downloads  chatbot.hgru + chatbot_vocab.json
#   Kaggle → files appear in Output tab (right panel)
#
# Copy both files into:
#   hypute-ml/examples/chatbot/
# then run:
#   ./build/chat examples/chatbot/chatbot.hgru examples/chatbot/chatbot_vocab.json
# ─────────────────────────────────────────────────────────────────────────────

# ── 0. Install (Colab already has torch; this just adds tqdm) ─────────────────
# !pip install tqdm -q

import re, json, struct, os, math, zipfile, tarfile, csv, io, urllib.request
import numpy as np
from collections import Counter

import torch
import torch.nn as nn

print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none — training will be slow")

# ── 1. Config ─────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 16000
EMBED_DIM   = 512
HIDDEN_DIM  = 1024
NUM_LAYERS  = 3
EPOCHS      = 20
SEQ_LEN     = 100
BATCH_SIZE  = 32      # increase to 64 if on A100
LR          = 0.001
CLIP        = 0.5
DROPOUT     = 0.3

SPECIAL = ['<pad>', '<unk>', '<eos>', '<user>', '<sys>']
PAD, UNK, EOS, USER, SYS = 0, 1, 2, 3, 4

# ── 2. Tokeniser ──────────────────────────────────────────────────────────────

def clean(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s'.,!?-]", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def words(text):
    return re.findall(r"[a-z0-9']+|[.,!?]", clean(text))

def build_vocab(dialogues):
    counter = Counter()
    for turns in dialogues:
        for t in turns:
            counter.update(words(t))
    vocab = {s: i for i, s in enumerate(SPECIAL)}
    for w, _ in counter.most_common(VOCAB_SIZE - len(SPECIAL)):
        vocab[w] = len(vocab)
    return vocab

def dialogue_to_ids(turns, vocab):
    ids = []
    for i, turn in enumerate(turns):
        ids.append(USER if i % 2 == 0 else SYS)
        ids.extend(vocab.get(w, UNK) for w in words(turn))
    ids.append(EOS)
    return ids

# ── 3. Dataset helpers ────────────────────────────────────────────────────────

def batchify(data, B):
    n = (len(data) // B) * B
    t = torch.tensor(data[:n], dtype=torch.long)
    return t.view(B, -1).t().contiguous()

def get_batch(src, i):
    l = min(SEQ_LEN, src.size(0) - 1 - i)
    return src[i:i+l], src[i+1:i+1+l].reshape(-1)

def detach(h):
    return [x.detach() for x in h]

# ── 4. Model ──────────────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding   = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD)
        self.drop        = nn.Dropout(DROPOUT)
        self.gru_cells   = nn.ModuleList(
            nn.GRUCell(EMBED_DIM if i == 0 else HIDDEN_DIM, HIDDEN_DIM)
            for i in range(NUM_LAYERS))
        self.output_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def forward(self, token_id, h_states):
        x = self.drop(self.embedding(token_id))
        new_h = []
        for i, cell in enumerate(self.gru_cells):
            x = cell(x, h_states[i])
            new_h.append(x)
            x = self.drop(x)
        return self.output_proj(x), new_h

    def init_hidden(self, B, device):
        return [torch.zeros(B, HIDDEN_DIM, device=device) for _ in self.gru_cells]

# ── 5. Export to .hgru ────────────────────────────────────────────────────────

HGRU_MAGIC, HGRU_VERSION = 0x48475255, 1

def qmatrix(w):
    rows, cols = w.shape
    q = np.zeros((rows, cols), np.int8)
    s = np.zeros(rows, np.float32)
    for i in range(rows):
        mx = float(np.abs(w[i]).max())
        if mx == 0:
            s[i] = 1.0
        else:
            s[i] = mx / 127.0
            q[i] = np.clip(np.round(w[i] / s[i]), -127, 127).astype(np.int8)
    return q, s

def write_qm(f, w_np):
    q, s = qmatrix(w_np.astype(np.float32))
    f.write(q.tobytes())
    f.write(s.tobytes())

def save_hgru(path, model):
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', HGRU_MAGIC, HGRU_VERSION))
        f.write(struct.pack('<4Q', VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS))
        write_qm(f, model.embedding.weight.detach().cpu().numpy())
        for cell in model.gru_cells:
            wih = cell.weight_ih.detach().cpu().numpy()
            whh = cell.weight_hh.detach().cpu().numpy()
            H = HIDDEN_DIM
            for mat in [wih[:H], wih[H:2*H], wih[2*H:],
                        whh[:H], whh[H:2*H], whh[2*H:]]:
                write_qm(f, mat)
        write_qm(f, model.output_proj.weight.detach().cpu().numpy())
    size_mb = os.path.getsize(path) / 1e6
    print(f"  Saved {path}  ({size_mb:.1f} MB)")

# ── 6. Data loaders ───────────────────────────────────────────────────────────

def load_cornell():
    url = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
    zpath = "cornell_dialogs.zip"
    ddir  = "cornell movie-dialogs corpus"
    if not os.path.exists(zpath):
        print("  Downloading Cornell Movie Dialogs …")
        urllib.request.urlretrieve(url, zpath)
    with zipfile.ZipFile(zpath) as zf:
        with zf.open(f"{ddir}/movie_lines.txt") as f:
            raw = f.read().decode("iso-8859-1")
        id2line = {}
        for row in raw.split("\n"):
            parts = row.split(" +++$+++ ")
            if len(parts) >= 5:
                id2line[parts[0].strip()] = parts[4].strip()
        with zf.open(f"{ddir}/movie_conversations.txt") as f:
            raw = f.read().decode("iso-8859-1")
    convs = []
    for row in raw.split("\n"):
        parts = row.split(" +++$+++ ")
        if len(parts) < 4: continue
        try:
            ids  = eval(parts[3].strip())
            conv = [id2line[i] for i in ids if i in id2line]
            if len(conv) >= 2: convs.append(conv)
        except Exception: continue
    print(f"  Cornell: {len(convs):,} conversations")
    return convs

def load_empathetic():
    url   = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
    tpath = "empatheticdialogues.tar.gz"
    if not os.path.exists(tpath):
        print("  Downloading EmpatheticDialogues …")
        urllib.request.urlretrieve(url, tpath)
    try:
        convs = []
        with tarfile.open(tpath, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".csv"): continue
                f = tar.extractfile(member)
                if not f: continue
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                cur_id, cur_turns = None, []
                for row in reader:
                    cid = row.get("conv_id", "")
                    utt = row.get("utterance", "").replace("_comma_", ",").strip()
                    if not utt: continue
                    if cid != cur_id:
                        if len(cur_turns) >= 2: convs.append(cur_turns)
                        cur_id, cur_turns = cid, []
                    cur_turns.append(utt)
                if len(cur_turns) >= 2: convs.append(cur_turns)
        print(f"  EmpatheticDialogues: {len(convs):,} conversations")
        return convs
    except Exception as e:
        print(f"  EmpatheticDialogues failed ({e}), skipping")
        return []

# ── 7. Train ──────────────────────────────────────────────────────────────────

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

print("\nLoading datasets …")
all_dialogues = load_cornell() + load_empathetic()
print(f"  Total: {len(all_dialogues):,} conversations")

print("\nBuilding vocabulary …")
vocab = build_vocab(all_dialogues)
with open('chatbot_vocab.json', 'w') as f:
    json.dump(vocab, f)
print(f"  Vocab: {len(vocab)} words")

all_ids = []
for dlg in all_dialogues:
    all_ids.extend(dialogue_to_ids(dlg, vocab))
print(f"  Tokens: {len(all_ids):,}")

split      = int(0.95 * len(all_ids))
train_data = batchify(all_ids[:split],       BATCH_SIZE).to(device)
val_data   = batchify(all_ids[split:], max(1, BATCH_SIZE // 4)).to(device)

model     = GRUModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss(ignore_index=PAD)

params = sum(p.numel() for p in model.parameters())
print(f"\n  Parameters: {params/1e6:.1f} M  ({params*4/1e6:.0f} MB fp32)")
print(f"  Training on {device} …\n")

best_val = float('inf')
for epoch in range(1, EPOCHS + 1):
    # ── train ──
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
            loss = loss + criterion(logits, targets[t*BATCH_SIZE:(t+1)*BATCH_SIZE])
        (loss / data.size(0)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()
        total_loss += loss.item() / data.size(0)
        steps += 1

    # ── validate ──
    model.eval()
    val_loss, vn = 0.0, 0
    vB = max(1, BATCH_SIZE // 4)
    hv = model.init_hidden(vB, device)
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, SEQ_LEN):
            d, t = get_batch(val_data, i)
            d, t = d.to(device), t.to(device)
            hv = detach(hv)
            for step in range(d.size(0)):
                logits, hv = model(d[step], hv)
            val_loss += criterion(logits, t[-vB:]).item()
            vn += 1

    val_ppl = math.exp(min(val_loss / max(vn, 1), 10))
    trn_ppl = math.exp(min(total_loss / steps, 10))
    scheduler.step(val_loss / max(vn, 1))

    mark = " ✓" if val_loss / max(vn, 1) < best_val else ""
    if val_loss / max(vn, 1) < best_val:
        best_val = val_loss / max(vn, 1)
        torch.save(model.state_dict(), 'chatbot_best.pt')

    print(f"  Epoch {epoch:2d}/{EPOCHS}  train_ppl {trn_ppl:6.1f}  val_ppl {val_ppl:6.1f}{mark}")

# ── 8. Export ─────────────────────────────────────────────────────────────────

print("\nExporting …")
model.load_state_dict(torch.load('chatbot_best.pt', map_location='cpu'))
model = model.cpu()
save_hgru('chatbot.hgru', model)
torch.save(model.state_dict(), 'chatbot_fp32.pt')
print("Done.\n")

# ── 9. Download ───────────────────────────────────────────────────────────────

try:
    # Colab
    from google.colab import files
    print("Downloading chatbot.hgru …")
    files.download('chatbot.hgru')
    print("Downloading chatbot_vocab.json …")
    files.download('chatbot_vocab.json')
    print("\nDone! Copy both files to:  hypute-ml/examples/chatbot/")
except ImportError:
    # Kaggle — files are in /kaggle/working/ automatically
    import shutil
    for fn in ['chatbot.hgru', 'chatbot_vocab.json', 'chatbot_fp32.pt']:
        if os.path.exists(fn):
            dst = f'/kaggle/working/{fn}'
            shutil.copy(fn, dst)
            print(f"  → {dst}  ({os.path.getsize(dst)/1e6:.1f} MB)")
    print("\nDone! Download from the Output tab on the right.")
