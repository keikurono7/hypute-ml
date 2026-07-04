"""
Train a GRU chatbot — v2: bigger model, more data.

Data sources (all direct HTTP, no HuggingFace):
  1. Cornell Movie Dialogs  — 83K conversations
  2. DailyDialog            — 13K everyday conversations
  3. EmpatheticDialogues    — 25K empathetic conversations

Model: vocab=16K, embed=512, hidden=1024, 3 layers

Run:  python train_chatbot.py
Out:  chatbot.hgru   (compressed weights)
      chatbot_vocab.json
      chatbot_fp32.pt
"""

import re, json, struct, os, math, zipfile, tarfile, csv, io
import urllib.request
import numpy as np
from collections import Counter

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

try:
    import torch, torch.nn as nn
except ImportError:
    raise SystemExit("pip install torch")

# ── Config ────────────────────────────────────────────────────────────────────
VOCAB_SIZE  = 16000
EMBED_DIM   = 512
HIDDEN_DIM  = 1024
NUM_LAYERS  = 3
EPOCHS      = 20
SEQ_LEN     = 100
BATCH_SIZE  = 16
LR          = 0.001
CLIP        = 0.5
DROPOUT     = 0.3

SPECIAL = ['<pad>', '<unk>', '<eos>', '<user>', '<sys>']
PAD, UNK, EOS, USER, SYS = 0, 1, 2, 3, 4

# ── Tokeniser ─────────────────────────────────────────────────────────────────

def clean(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s'.,!?-]", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

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

# ── Dataset ───────────────────────────────────────────────────────────────────

def batchify(data, batch_size):
    n = (len(data) // batch_size) * batch_size
    t = torch.tensor(data[:n], dtype=torch.long)
    return t.view(batch_size, -1).t().contiguous()

def get_batch(src, i):
    l = min(SEQ_LEN, src.size(0) - 1 - i)
    return src[i:i+l], src[i+1:i+1+l].reshape(-1)

def detach(h): return [x.detach() for x in h]

# ── Model ─────────────────────────────────────────────────────────────────────

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

# ── Export ────────────────────────────────────────────────────────────────────

HGRU_MAGIC, HGRU_VERSION = 0x48475255, 1

def qmatrix(w):
    rows, cols = w.shape
    q, s = np.zeros((rows, cols), np.int8), np.zeros(rows, np.float32)
    for i in range(rows):
        mx = float(np.abs(w[i]).max())
        if mx == 0: s[i] = 1.0
        else:
            s[i] = mx / 127.0
            q[i] = np.clip(np.round(w[i] / s[i]), -127, 127).astype(np.int8)
    return q, s

def write_qm(f, w_np):
    q, s = qmatrix(w_np.astype(np.float32))
    f.write(q.tobytes()); f.write(s.tobytes())

def save_hgru(path, model):
    with open(path, 'wb') as f:
        f.write(struct.pack('<II', HGRU_MAGIC, HGRU_VERSION))
        f.write(struct.pack('<4Q', VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS))
        write_qm(f, model.embedding.weight.detach().cpu().numpy())
        for i, cell in enumerate(model.gru_cells):
            wih = cell.weight_ih.detach().cpu().numpy()
            whh = cell.weight_hh.detach().cpu().numpy()
            H = HIDDEN_DIM
            for mat in [wih[:H], wih[H:2*H], wih[2*H:],
                        whh[:H], whh[H:2*H], whh[2*H:]]:
                write_qm(f, mat)
        write_qm(f, model.output_proj.weight.detach().cpu().numpy())
    print(f"  Saved {path}  ({os.path.getsize(path)/1e6:.1f} MB int8)")

# ── Data loaders (all direct HTTP, no HuggingFace) ────────────────────────────

CORNELL_URL = "https://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
CORNELL_ZIP = "cornell_dialogs.zip"
CORNELL_DIR = "cornell movie-dialogs corpus"

def load_cornell():
    if not os.path.exists(CORNELL_ZIP):
        print("  Downloading Cornell Movie Dialogs …")
        urllib.request.urlretrieve(CORNELL_URL, CORNELL_ZIP)
    with zipfile.ZipFile(CORNELL_ZIP) as zf:
        with zf.open(f"{CORNELL_DIR}/movie_lines.txt") as f:
            raw = f.read().decode("iso-8859-1")
        id2line = {}
        for row in raw.split("\n"):
            parts = row.split(" +++$+++ ")
            if len(parts) >= 5:
                id2line[parts[0].strip()] = parts[4].strip()
        with zf.open(f"{CORNELL_DIR}/movie_conversations.txt") as f:
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

DAILY_URL = "http://yanran.li/files/ijcnlp_dailydialog.zip"
DAILY_ZIP = "dailydialog.zip"

def load_dailydialog():
    if not os.path.exists(DAILY_ZIP):
        print("  Downloading DailyDialog …")
        try:
            urllib.request.urlretrieve(DAILY_URL, DAILY_ZIP)
        except Exception as e:
            print(f"  DailyDialog download failed ({e}), skipping")
            return []
    try:
        convs = []
        with zipfile.ZipFile(DAILY_ZIP) as zf:
            for name in zf.namelist():
                if "dialogues_text" in name:
                    with zf.open(name) as f:
                        for line in f.read().decode("utf-8").split("\n"):
                            line = line.strip()
                            if not line: continue
                            turns = [t.strip() for t in line.split("__eou__") if t.strip()]
                            if len(turns) >= 2:
                                convs.append(turns)
        print(f"  DailyDialog: {len(convs):,} conversations")
        return convs
    except Exception as e:
        print(f"  DailyDialog parse failed ({e}), skipping")
        return []

EMPATH_URL = "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"
EMPATH_TAR = "empatheticdialogues.tar.gz"

def load_empathetic():
    if not os.path.exists(EMPATH_TAR):
        print("  Downloading EmpatheticDialogues …")
        try:
            urllib.request.urlretrieve(EMPATH_URL, EMPATH_TAR)
        except Exception as e:
            print(f"  EmpatheticDialogues download failed ({e}), skipping")
            return []
    try:
        convs = []
        with tarfile.open(EMPATH_TAR, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.endswith(".csv"): continue
                f = tar.extractfile(member)
                if not f: continue
                reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
                current_id, current_turns = None, []
                for row in reader:
                    cid = row.get("conv_id", "")
                    utt = row.get("utterance", "").replace("_comma_", ",").strip()
                    if not utt: continue
                    if cid != current_id:
                        if len(current_turns) >= 2: convs.append(current_turns)
                        current_id, current_turns = cid, []
                    current_turns.append(utt)
                if len(current_turns) >= 2: convs.append(current_turns)
        print(f"  EmpatheticDialogues: {len(convs):,} conversations")
        return convs
    except Exception as e:
        print(f"  EmpatheticDialogues parse failed ({e}), skipping")
        return []

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading datasets …")
    all_dialogues = load_cornell() + load_dailydialog() + load_empathetic()
    print(f"  Total: {len(all_dialogues):,} conversations")

    print("Building vocabulary …")
    vocab = build_vocab(all_dialogues)
    id2word = {v: k for k, v in vocab.items()}
    with open('chatbot_vocab.json', 'w') as f:
        json.dump(vocab, f)
    print(f"  Vocab: {len(vocab)} words")

    all_ids = []
    for dlg in all_dialogues:
        all_ids.extend(dialogue_to_ids(dlg, vocab))
    print(f"  Total tokens: {len(all_ids):,}")

    split = int(0.95 * len(all_ids))
    train_data = batchify(all_ids[:split],       BATCH_SIZE).to(device)
    val_data   = batchify(all_ids[split:], max(1, BATCH_SIZE // 4)).to(device)

    model     = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params/1e6:.1f} M  ({params*4/1e6:.0f} MB fp32)\n")

    best_val = float('inf')
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
                loss = loss + criterion(logits,
                    targets[t*BATCH_SIZE:(t+1)*BATCH_SIZE])
            (loss / data.size(0)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step()
            total_loss += loss.item() / data.size(0)
            steps += 1

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

        val_ppl = math.exp(min(val_loss / max(vn,1), 10))
        trn_ppl = math.exp(min(total_loss / steps, 10))
        scheduler.step(val_loss / max(vn, 1))

        mark = " *" if val_loss / max(vn,1) < best_val else ""
        if val_loss / max(vn,1) < best_val:
            best_val = val_loss / max(vn,1)
            torch.save(model.state_dict(), 'chatbot_best.pt')

        print(f"  Epoch {epoch:2d}/{EPOCHS}  train_ppl {trn_ppl:6.1f}  "
              f"val_ppl {val_ppl:6.1f}{mark}")

    print("\nExporting best checkpoint …")
    model.load_state_dict(torch.load('chatbot_best.pt', map_location='cpu'))
    model = model.cpu()
    save_hgru('chatbot.hgru', model)
    torch.save(model.state_dict(), 'chatbot_fp32.pt')
    print(f"  Saved chatbot_fp32.pt")
    print(f"\nDone. Run: ./chat chatbot.hgru chatbot_vocab.json")

if __name__ == '__main__':
    main()
