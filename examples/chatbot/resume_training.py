"""Resume v1 chatbot training from epoch 16 checkpoint → run epochs 17-25."""

import re, json, struct, os, math, zipfile, urllib.request
import numpy as np
from collections import Counter
import torch, torch.nn as nn

VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS = 8000, 256, 512, 2
START_EPOCH, EPOCHS = 17, 25
SEQ_LEN, BATCH_SIZE, LR, CLIP, DROPOUT = 80, 32, 0.002, 0.5, 0.35
SPECIAL = ['<pad>', '<unk>', '<eos>', '<user>', '<sys>']
PAD, UNK, EOS, USER, SYS = 0, 1, 2, 3, 4
HGRU_MAGIC, HGRU_VERSION = 0x48475255, 1

def clean(t): return re.sub(r'\s+',' ',re.sub(r"[^a-z0-9\s'.,!?-]",' ',t.lower().strip())).strip()
def words(t): return re.findall(r"[a-z0-9']+|[.,!?]", clean(t))
def build_vocab(dialogues):
    counter = Counter()
    for turns in dialogues:
        for t in turns: counter.update(words(t))
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
def batchify(data, B):
    n = (len(data)//B)*B; t = torch.tensor(data[:n], dtype=torch.long)
    return t.view(B,-1).t().contiguous()
def get_batch(src, i):
    l = min(SEQ_LEN, src.size(0)-1-i); return src[i:i+l], src[i+1:i+1+l].reshape(-1)
def detach(h): return [x.detach() for x in h]

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding   = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=PAD)
        self.drop        = nn.Dropout(DROPOUT)
        self.gru_cells   = nn.ModuleList(nn.GRUCell(EMBED_DIM if i==0 else HIDDEN_DIM, HIDDEN_DIM) for i in range(NUM_LAYERS))
        self.output_proj = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)
    def forward(self, token_id, h_states):
        x = self.drop(self.embedding(token_id)); new_h = []
        for i, cell in enumerate(self.gru_cells):
            x = cell(x, h_states[i]); new_h.append(x); x = self.drop(x)
        return self.output_proj(x), new_h
    def init_hidden(self, B, device):
        return [torch.zeros(B, HIDDEN_DIM, device=device) for _ in self.gru_cells]

def qmatrix(w):
    rows, cols = w.shape; q = np.zeros((rows,cols),np.int8); s = np.zeros(rows,np.float32)
    for i in range(rows):
        mx = float(np.abs(w[i]).max())
        if mx==0: s[i]=1.0
        else: s[i]=mx/127.0; q[i]=np.clip(np.round(w[i]/s[i]),-127,127).astype(np.int8)
    return q, s
def write_qm(f, w): q,s=qmatrix(w.astype(np.float32)); f.write(q.tobytes()); f.write(s.tobytes())
def save_hgru(path, model):
    with open(path,'wb') as f:
        f.write(struct.pack('<II', HGRU_MAGIC, HGRU_VERSION))
        f.write(struct.pack('<4Q', VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS))
        write_qm(f, model.embedding.weight.detach().cpu().numpy())
        for cell in model.gru_cells:
            wih = cell.weight_ih.detach().cpu().numpy()
            whh = cell.weight_hh.detach().cpu().numpy(); H = HIDDEN_DIM
            for mat in [wih[:H],wih[H:2*H],wih[2*H:],whh[:H],whh[H:2*H],whh[2*H:]]:
                write_qm(f, mat)
        write_qm(f, model.output_proj.weight.detach().cpu().numpy())
    print(f"  Saved {path}  ({os.path.getsize(path)/1e6:.1f} MB int8)")

def load_cornell():
    CORNELL_ZIP = "cornell_dialogs.zip"; CORNELL_DIR = "cornell movie-dialogs corpus"
    with zipfile.ZipFile(CORNELL_ZIP) as zf:
        with zf.open(f"{CORNELL_DIR}/movie_lines.txt") as f:
            raw = f.read().decode("iso-8859-1")
        id2line = {}
        for row in raw.split("\n"):
            parts = row.split(" +++$+++ ")
            if len(parts) >= 5: id2line[parts[0].strip()] = parts[4].strip()
        with zf.open(f"{CORNELL_DIR}/movie_conversations.txt") as f:
            raw = f.read().decode("iso-8859-1")
    convs = []
    for row in raw.split("\n"):
        parts = row.split(" +++$+++ ")
        if len(parts) < 4: continue
        try:
            ids = eval(parts[3].strip()); conv = [id2line[i] for i in ids if i in id2line]
            if len(conv) >= 2: convs.append(conv)
        except: pass
    return convs

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Resuming from epoch {START_EPOCH}")

    all_dialogues = load_cornell()
    vocab = build_vocab(all_dialogues)
    with open('chatbot_vocab.json', 'w') as f: json.dump(vocab, f)
    all_ids = []
    for dlg in all_dialogues: all_ids.extend(dialogue_to_ids(dlg, vocab))
    split = int(0.95 * len(all_ids))
    train_data = batchify(all_ids[:split], BATCH_SIZE).to(device)
    val_data   = batchify(all_ids[split:], max(1, BATCH_SIZE//4)).to(device)

    model = GRUModel().to(device)
    model.load_state_dict(torch.load('chatbot_best.pt', map_location=device))
    print(f"  Loaded chatbot_best.pt (epoch {START_EPOCH-1} weights)")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR * 0.5)  # lower lr for fine-tuning
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best_val = float('inf')
    for epoch in range(START_EPOCH, EPOCHS + 1):
        model.train()
        total_loss, steps = 0.0, 0
        h = model.init_hidden(BATCH_SIZE, device)
        for i in range(0, train_data.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(train_data, i)
            data, targets = data.to(device), targets.to(device)
            h = detach(h); optimizer.zero_grad()
            loss = torch.tensor(0.0, device=device)
            for t in range(data.size(0)):
                logits, h = model(data[t], h)
                loss = loss + criterion(logits, targets[t*BATCH_SIZE:(t+1)*BATCH_SIZE])
            (loss / data.size(0)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            optimizer.step(); total_loss += loss.item() / data.size(0); steps += 1

        model.eval(); val_loss, vn = 0.0, 0
        vB = max(1, BATCH_SIZE//4); hv = model.init_hidden(vB, device)
        with torch.no_grad():
            for i in range(0, val_data.size(0) - 1, SEQ_LEN):
                d, t = get_batch(val_data, i); d, t = d.to(device), t.to(device)
                hv = detach(hv)
                for step in range(d.size(0)): logits, hv = model(d[step], hv)
                val_loss += criterion(logits, t[-vB:]).item(); vn += 1

        val_ppl = math.exp(min(val_loss/max(vn,1), 10))
        trn_ppl = math.exp(min(total_loss/steps, 10))
        mark = " *" if val_loss/max(vn,1) < best_val else ""
        if val_loss/max(vn,1) < best_val:
            best_val = val_loss/max(vn,1)
            torch.save(model.state_dict(), 'chatbot_best.pt')
        print(f"  Epoch {epoch:2d}/{EPOCHS}  train_ppl {trn_ppl:6.1f}  val_ppl {val_ppl:6.1f}{mark}")

    print("\nExporting …")
    model.load_state_dict(torch.load('chatbot_best.pt', map_location='cpu'))
    save_hgru('chatbot.hgru', model.cpu())
    torch.save(model.state_dict(), 'chatbot_fp32.pt')
    print("Done.")

if __name__ == '__main__':
    main()
