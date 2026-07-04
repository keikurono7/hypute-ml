"""
Side-by-side comparison: our int8 GRU chatbot vs DialoGPT-small (117M, fp32).

Runs the same 10 conversation turns through both models, then prints:
  - Response quality (text side by side)
  - Avg response latency
  - Memory footprint

Prerequisites: train_chatbot.py must have produced chatbot.hgru + chatbot_vocab.json.
Run: python compare_transformer.py
"""

import json, re, os, time, struct, math
import numpy as np

try:
    import torch, torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise SystemExit("pip install torch transformers")

# ── Prompts used for both models ──────────────────────────────────────────────

PROMPTS = [
    "hello how are you",
    "what do you do for a living",
    "do you like movies",
    "what is your favorite food",
    "tell me something interesting",
    "how was your day",
    "do you have any hobbies",
    "what do you think about technology",
    "i am feeling a bit tired today",
    "thanks for the conversation",
]

# ══════════════════════════════════════════════════════════════════════════════
# Our GRU chatbot (int8, loaded via NumPy — mirrors the C++ engine exactly)
# ══════════════════════════════════════════════════════════════════════════════

VOCAB_SIZE  = 8000
EMBED_DIM   = 256
HIDDEN_DIM  = 512
NUM_LAYERS  = 2
HGRU_MAGIC  = 0x48475255

def read_qmatrix(f, rows, cols):
    q = np.frombuffer(f.read(rows * cols), dtype=np.int8).reshape(rows, cols).astype(np.float32)
    s = np.frombuffer(f.read(rows * 4), dtype=np.float32)
    return q * s[:, None]

def load_hgru(path):
    with open(path, 'rb') as f:
        import struct as st
        magic, ver = st.unpack('<II', f.read(8))
        assert magic == HGRU_MAGIC, "Not a .hgru file"
        V, E, H, L = st.unpack('<4Q', f.read(32))
        emb = read_qmatrix(f, V, E)
        cells = []
        for i in range(L):
            in_d = E if i == 0 else H
            W_r = read_qmatrix(f, H, in_d);  W_z = read_qmatrix(f, H, in_d)
            W_n = read_qmatrix(f, H, in_d);  U_r = read_qmatrix(f, H, H)
            U_z = read_qmatrix(f, H, H);     U_n = read_qmatrix(f, H, H)
            cells.append((W_r, W_z, W_n, U_r, U_z, U_n))
        out_proj = read_qmatrix(f, V, H)
    return emb, cells, out_proj

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def gru_step(x, h, cell):
    W_r, W_z, W_n, U_r, U_z, U_n = cell
    r = sigmoid(W_r @ x + U_r @ h)
    z = sigmoid(W_z @ x + U_z @ h)
    n = np.tanh(W_n @ x + r * (U_n @ h))
    return (1 - z) * n + z * h

def words(text):
    return re.findall(r"[a-z0-9']+|[.,!?]", re.sub(r"[^a-z0-9\s'.,!?-]", ' ',
                      text.lower().strip()))

def topk_sample(logits, temperature=0.75, top_k=40):
    idx = np.argpartition(logits, -top_k)[-top_k:]
    lg  = logits[idx] / temperature
    lg -= lg.max()
    p   = np.exp(lg); p /= p.sum()
    return int(np.random.choice(idx, p=p))

class OurChatbot:
    def __init__(self, hgru_path, vocab_path):
        self.emb, self.cells, self.out = load_hgru(hgru_path)
        with open(vocab_path) as f:
            self.w2id = json.load(f)
        self.id2w = {v: k for k, v in self.w2id.items()}
        self.USER = self.w2id.get('<user>', 3)
        self.SYS  = self.w2id.get('<sys>',  4)
        self.EOS  = self.w2id.get('<eos>',  2)
        self.h = [np.zeros(HIDDEN_DIM) for _ in self.cells]
        self.last_logits = np.zeros(VOCAB_SIZE)

    def _step(self, token_id):
        idx = int(token_id) % VOCAB_SIZE
        x = self.emb[idx].copy()
        for i, cell in enumerate(self.cells):
            self.h[i] = gru_step(x, self.h[i], cell)
            x = self.h[i]
        self.last_logits = self.out @ x

    def reset(self):
        self.h = [np.zeros(HIDDEN_DIM) for _ in self.cells]

    def respond(self, text, max_tokens=60, temperature=0.75, top_k=40):
        self._step(self.USER)
        for w in words(text):
            self._step(self.w2id.get(w, 1))
        self._step(self.SYS)

        tokens = []
        for _ in range(max_tokens):
            nid = topk_sample(self.last_logits, temperature, top_k)
            if nid in (self.USER, self.EOS): break
            w = self.id2w.get(nid, '')
            if w and not w.startswith('<'):
                tokens.append(w)
            self._step(nid)

        resp = ''
        for i, w in enumerate(tokens):
            resp += ('' if i == 0 or w in '.,!?' else ' ') + w
        if resp:
            resp = resp[0].upper() + resp[1:]
            if resp[-1] not in '.!?': resp += '.'
        return resp or '...'

# ══════════════════════════════════════════════════════════════════════════════
# DialoGPT-small (117M params, fp32)
# ══════════════════════════════════════════════════════════════════════════════

class DialoGPTBot:
    MODEL = "microsoft/DialoGPT-small"

    def __init__(self):
        print(f"  Loading {self.MODEL} …")
        self.tok   = AutoTokenizer.from_pretrained(self.MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL)
        self.model.eval()
        self.history = None

    def reset(self):
        self.history = None

    def respond(self, text, max_new_tokens=60):
        inp = self.tok.encode(text + self.tok.eos_token, return_tensors='pt')
        if self.history is not None:
            inp = torch.cat([self.history, inp], dim=-1)
        with torch.no_grad():
            out = self.model.generate(
                inp,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tok.eos_token_id,
                do_sample=True,
                top_k=40,
                temperature=0.75,
            )
        self.history = out
        response = self.tok.decode(
            out[:, inp.shape[-1]:][0], skip_special_tokens=True)
        return response.strip() or '...'

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run_bot(bot, prompts):
    responses, latencies = [], []
    bot.reset()
    for p in prompts:
        t0 = time.perf_counter()
        r  = bot.respond(p)
        latencies.append((time.perf_counter() - t0) * 1000)
        responses.append(r)
    return responses, latencies

def main():
    for f in ('chatbot.hgru', 'chatbot_vocab.json'):
        if not os.path.exists(f):
            raise SystemExit(f"Missing {f} — run train_chatbot.py first")

    print("\n[1/2] Loading our GRU chatbot …")
    our_bot = OurChatbot('chatbot.hgru', 'chatbot_vocab.json')
    our_size_mb = os.path.getsize('chatbot.hgru') / 1e6

    print("[2/2] Loading DialoGPT-small …")
    gpt_bot = DialoGPTBot()
    gpt_params  = sum(p.numel() for p in gpt_bot.model.parameters())
    gpt_size_mb = gpt_params * 4 / 1e6

    print("\nRunning conversations …\n")
    our_resp, our_lat = run_bot(our_bot, PROMPTS)
    gpt_resp, gpt_lat = run_bot(gpt_bot, PROMPTS)

    # ── Side-by-side responses ────────────────────────────────────────────────
    W = 74
    print("=" * W)
    print(f"  {'CONVERSATION COMPARISON':^{W-2}}")
    print("=" * W)
    col = (W - 4) // 2

    def wrap(text, width):
        words_l = text.split()
        lines, line = [], ''
        for w in words_l:
            if len(line) + len(w) + 1 <= width:
                line += (' ' if line else '') + w
            else:
                lines.append(line); line = w
        if line: lines.append(line)
        return lines

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  You: {prompt}")
        print(f"  {'─'*col}  {'─'*col}")
        print(f"  {'Ours (GRU int8)':<{col}}  {'DialoGPT-small (fp32)':<{col}}")
        print(f"  {'─'*col}  {'─'*col}")
        ow = wrap(our_resp[i], col)
        gw = wrap(gpt_resp[i], col)
        for j in range(max(len(ow), len(gw))):
            ol = ow[j] if j < len(ow) else ''
            gl = gw[j] if j < len(gw) else ''
            print(f"  {ol:<{col}}  {gl:<{col}}")

    # ── Stats table ───────────────────────────────────────────────────────────
    print("\n" + "=" * W)
    print(f"  {'STATS COMPARISON':^{W-2}}")
    print("=" * W)
    print(f"  {'Metric':<30}  {'Ours (GRU)':>12}  {'DialoGPT-s':>12}")
    print(f"  {'─'*30}  {'─'*12}  {'─'*12}")
    print(f"  {'Parameters':<30}  {sum(c[0].size for c in our_bot.cells)*6/1e6 + VOCAB_SIZE*(EMBED_DIM+HIDDEN_DIM)/1e6:>11.1f}M  {gpt_params/1e6:>11.0f}M")
    print(f"  {'Model size':<30}  {our_size_mb:>10.1f}MB  {gpt_size_mb:>10.0f}MB")
    print(f"  {'Avg response latency':<30}  {sum(our_lat)/len(our_lat):>9.0f}ms  {sum(gpt_lat)/len(gpt_lat):>9.0f}ms")
    print(f"  {'Size ratio':<30}  {'1×':>12}  {gpt_size_mb/our_size_mb:>11.0f}×")
    print(f"  {'Speed ratio':<30}  {'1×':>12}  {sum(gpt_lat)/sum(our_lat):>10.1f}×")
    print("=" * W)
    print(f"""
  Takeaway
    DialoGPT gives more natural, context-aware responses.
    Our GRU runs in {our_size_mb:.0f} MB vs {gpt_size_mb:.0f} MB — {gpt_size_mb/our_size_mb:.0f}× smaller.
    On a device with 64 MB RAM budget, only our model fits.
""")


if __name__ == '__main__':
    main()
