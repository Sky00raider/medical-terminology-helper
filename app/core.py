"""Medical terminology GPT, written from scratch: corpus, BPE, attention, training, generation, study tools.

One section per milestone, in order. The model code is plain PyTorch (no nn.MultiheadAttention, no pretrained weights).
"""
import json
import math
import random
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Milestone 1: Curate a small corpus of public domain medical texts (e.g., anatomy, physiology).
# Hint: Data collection, text cleaning.
_FOLD = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-",
         "æ": "ae", "œ": "oe", "Æ": "Ae", "Œ": "Oe", "°": " degrees ", "½": "1/2"}
_BRACKET = re.compile(r"\[[^\]]{0,200}\](?!\.)")     # [Illustration], [Footnote 1], [Pg 12]; keeps "[etymology]." forms
_FIG = re.compile(r"\(\s*(?:see\s+)?Figs?\.?\s*[\d\s,\-and.]+\)|\bFigs?\.\s*\d+[\d\s,\-and]*\.?", re.I)


def clean_text(raw: str) -> str:
    """Raw scanned text to clean paragraphs separated by blank lines (ASCII only, one paragraph per line).

    Folds typography to ASCII, strips accents, rejoins words hyphenated across line breaks, drops figure references,
    page furniture and paragraphs that are mostly OCR noise, and removes duplicate paragraphs."""
    text = "".join(_FOLD.get(c, c) for c in raw.replace("\r", ""))
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    text = re.sub(r"[^\x20-\x7e\n]", " ", text)
    text = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", text)   # re-join "Phe-\nnylacetamid"
    text = _BRACKET.sub(" ", text)
    text = _FIG.sub(" ", text)
    seen, out = set(), []
    for block in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", block).strip()
        if is_clean_paragraph(para) and para not in seen:
            seen.add(para)
            out.append(para)
    return "\n\n".join(out)


def is_clean_paragraph(p: str, min_words: int = 6, min_chars: int = 40) -> bool:
    """Reject headings, page numbers, index lines and OCR garbage."""
    words = p.split()
    if len(words) < min_words or len(p) < min_chars:
        return False
    letters = sum(c.isalpha() for c in p)
    if letters / len(p) < 0.75:                                   # tables, formulas, numbers
        return False
    plain = sum(bool(re.fullmatch(r"[A-Za-z][a-z]*[,.;:)]?|[A-Za-z]{1,3}", w)) for w in words)
    if plain / len(words) < 0.8:                                  # OCR noise: odd capitals, digits inside words
        return False
    if sum(len(w) for w in words) / len(words) > 9 or len(words) >= 8 and sum(len(w) == 1 for w in words) / len(words) > 0.15:
        return False
    upper = sum(c.isupper() for c in p) / letters
    return upper < 0.3                                            # running heads and chapter titles in capitals


def chunk_paragraph(p: str, max_chars: int = 900) -> list[str]:
    """Split a long paragraph at sentence ends so the corpus has uniformly sized units to shuffle and split."""
    if len(p) <= max_chars:
        return [p]
    out, cur = [], ""
    for sent in re.split(r"(?<=[.;?!]) +", p):
        if cur and len(cur) + len(sent) > max_chars:
            out.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    return out + [cur] if cur else out


_ENTRY = re.compile(r"^([A-Z][A-Za-z'\-]+(?: [A-Za-z'\-]+){0,2}) ?(?:[({][^()\[\]{}]{2,60}[)}])?\s*(?:\[[^\]]{0,150}\])?[^.]{0,40}?\.\s+([A-Z].{14,})$")
_NOT_A_TERM = set("the a an in of on it this that these those there when if as by for with from at is are was were has have "
                  "be to and or but not no we they he she his her its which who what where how".split())


def shorten_definition(d: str, max_chars: int = 300) -> str:
    """Keep the leading sentences of an entry (up to max_chars) and drop the run-on sub-entries ('A. Nerve, ...')."""
    d = re.sub(r"\bI\. ", "1. ", d.strip())
    out = ""
    for sent in re.split(r"(?<=[.;]) +(?=[A-Z0-9])", d):
        if out and (len(out) + len(sent) > max_chars or re.match(r"[A-Z]\.( |$)", sent)):
            break
        out = f"{out} {sent}".strip()
    return out if len(out) <= max_chars else ""  # a single over-long sentence would be cut mid-way: skip it


def parse_dictionary(raw: str) -> list[tuple[str, str]]:
    """Pull (TERM, definition) pairs out of a scanned medical dictionary, where an entry reads
    'Acetate (as-et-at) [acetum, vinegar]. Any salt of acetic acid.'; the pronunciation and etymology are dropped.
    Cross-references ('See ...'), repeated terms and OCR-damaged entries are skipped."""
    text = "".join(_FOLD.get(c, c) for c in raw.replace("\r", ""))
    text = re.sub(r"[^\x20-\x7e\n]", " ", text)
    text = re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", text)
    entries, seen = [], set()
    for block in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", block).strip()
        m = _ENTRY.match(para)
        if not m:
            continue
        term, definition = m.group(1), shorten_definition(m.group(2))
        if any(w.lower() in _NOT_A_TERM for w in term.split()) or term.lower() in seen or definition.startswith("See "):
            continue
        if not is_clean_paragraph(definition, min_words=3, min_chars=20):
            continue
        seen.add(term.lower())
        entries.append((term.upper(), definition))
    return entries


def format_entry(term: str, definition: str) -> str:
    return f"{term}. {definition}"


GLOSSARY_PATH = "data/glossary.json"


def load_glossary(path=GLOSSARY_PATH) -> list[tuple[str, str]]:
    """Held-out dictionary entries [[term, definition], ...] written by scripts/build_corpus.py."""
    return [tuple(e) for e in json.loads(Path(path).read_text(encoding="utf-8"))]


def split_corpus(paragraphs: list[str], val_frac: float = 0.06, seed: int = 0) -> tuple[list[str], list[str]]:
    """Shuffle (seeded) and hold out a share of paragraphs, so topics mix and the model never sees the held-out ones."""
    paras = list(paragraphs)
    random.Random(seed).shuffle(paras)
    cut = max(1, int(len(paras) * val_frac))
    return paras[cut:], paras[:cut]


def corpus_stats(text: str) -> dict:
    words = re.findall(r"[A-Za-z]+", text)
    return {"characters": len(text), "words": len(words), "unique_words": len({w.lower() for w in words}),
            "paragraphs": text.count("\n\n") + 1, "alphabet": len(set(text))}


# --- Milestone 2: Implement the byte-pair tokenizer and generate embeddings from the medical corpus.
# Hint: Byte-Pair Encoding (BPE), embedding layer.

# A chunk is a word with its leading space, a run of digits, a run of punctuation, or whitespace.
# Merges never cross a chunk boundary, so a token is always part of one word.
CHUNK_RE = re.compile(r" ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+")


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every adjacent occurrence of `pair` in `ids` with `new_id`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """Byte-pair encoding over characters: start from one token per character, then repeatedly merge the most
    frequent adjacent pair. Frequent medical stems and endings (-itis, -ectomy, cardi-, ...) become single tokens."""

    def __init__(self, chars: list[str], merges: list[tuple[int, int]]):
        self.chars = chars
        self.merges = [tuple(m) for m in merges]  # merge k creates token id len(chars) + k
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = dict(enumerate(chars))
        for k, (a, b) in enumerate(self.merges):
            self.itos[len(chars) + k] = self.itos[a] + self.itos[b]
        self.ranks = {pair: len(chars) + k for k, pair in enumerate(self.merges)}  # lower id = learned earlier
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int = 1024) -> "BPETokenizer":
        chars = sorted(set(text))
        stoi = {c: i for i, c in enumerate(chars)}
        freqs = Counter(CHUNK_RE.findall(text))  # work on unique chunks weighted by count, not on the raw stream
        words = [[stoi[c] for c in w] for w in freqs]
        counts = list(freqs.values())
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        where: dict[tuple[int, int], set[int]] = defaultdict(set)
        for w, (ids, f) in enumerate(zip(words, counts)):
            for pair in zip(ids, ids[1:]):
                pair_counts[pair] += f
                where[pair].add(w)
        merges: list[tuple[int, int]] = []
        for k in range(max(vocab_size - len(chars), 0)):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            merges.append(best)
            new_id = len(chars) + k
            for w in where.pop(best):  # only the chunks containing the pair change
                ids, f = words[w], counts[w]
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] -= f
                ids = words[w] = merge_pair(ids, best, new_id)
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += f
                    where[pair].add(w)
            for pair in [p for p, c in pair_counts.items() if c <= 0]:
                del pair_counts[pair]
        return cls(chars, merges)

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + len(self.merges)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk not in self._cache:
            ids = [self.stoi[c] for c in chunk]
            while len(ids) > 1:
                pair = min(zip(ids, ids[1:]), key=lambda p: self.ranks.get(p, float("inf")))
                if pair not in self.ranks:
                    break
                ids = merge_pair(ids, pair, self.ranks[pair])
            self._cache[chunk] = ids
        return self._cache[chunk]

    def encode(self, s: str, skip_unknown: bool = False) -> list[int]:
        """Text to ids. Characters unseen in training raise ValueError, or are dropped with skip_unknown=True."""
        unknown = set(s) - self.stoi.keys()
        if unknown:
            if not skip_unknown:
                raise ValueError(f"characters not in vocabulary: {sorted(unknown)}")
            s = "".join(c for c in s if c in self.stoi)
        out: list[int] = []
        for chunk in CHUNK_RE.findall(s):
            out.extend(self._encode_chunk(chunk))
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def tokens(self, s: str) -> list[str]:
        """The token strings for `s`, e.g. ' gastro', 'cnem', 'ius'. Handy for seeing how a term is split."""
        return [self.itos[i] for i in self.encode(s, skip_unknown=True)]

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"chars": self.chars, "merges": self.merges}), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["chars"], d["merges"])


class Embeddings(nn.Module):
    """Token embedding (what the token is) plus learned position embedding (where it is): (B, T) -> (B, T, C)."""

    def __init__(self, vocab_size: int, n_embd: int, block_size: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)

    def forward(self, idx):
        return self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))


def embed_text(tok: BPETokenizer, emb: Embeddings, text: str) -> torch.Tensor:
    """Encode `text` and return its (1, T, n_embd) embeddings."""
    ids = torch.tensor([tok.encode(text, skip_unknown=True)])
    with torch.no_grad():
        return emb(ids)


def nearest_tokens(tok: BPETokenizer, emb_weight: torch.Tensor, token: str, k: int = 8) -> list[tuple[str, float]]:
    """Tokens whose learned embeddings point the same way as `token` (cosine similarity)."""
    inv = {t: i for i, t in tok.itos.items()}
    ids = [inv[token]] if token in inv else tok.encode(token, skip_unknown=True)
    if len(ids) != 1:
        raise ValueError(f"{token!r} is not a single token (it splits into {tok.tokens(token)})")
    w = F.normalize(emb_weight.detach(), dim=-1)
    sims = w @ w[ids[0]]
    sims[ids[0]] = -2
    top = torch.topk(sims, k)
    return [(tok.itos[int(i)], float(s)) for s, i in zip(top.values, top.indices)]


# --- Milestone 3: Build the causal self-attention mechanism and assemble the transformer block.
# Hint: Causal self-attention, transformer block assembly.

@dataclass
class Config:
    vocab_size: int = 1024
    block_size: int = 128   # context window in tokens
    n_embd: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    """Masked multi-head scaled dot-product attention. Position t may only look at positions <= t."""

    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # queries, keys, values for every head at once
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.drop_attn = nn.Dropout(cfg.dropout)
        self.drop_out = nn.Dropout(cfg.dropout)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)), persistent=False)

    def forward(self, x, return_weights: bool = False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q, k, v = (t.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2) for t in (q, k, v))  # (B, h, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5          # every pair of positions, scaled
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # hide the future
        wei = F.softmax(wei, dim=-1)                                 # each row sums to 1
        out = self.drop_attn(wei) @ v                                # (B, h, T, hs)
        out = self.drop_out(self.proj(out.transpose(1, 2).reshape(B, T, C)))
        return (out, wei) if return_weights else out


class FeedForward(nn.Module):
    """Per-token MLP: expand 4x, GELU, project back. Attention mixes tokens; this transforms each one."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
                                 nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: pre-norm attention and feed-forward, each with a residual connection."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.n_embd), nn.LayerNorm(cfg.n_embd)
        self.attn, self.ff = CausalSelfAttention(cfg), FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.emb = Embeddings(cfg.vocab_size, cfg.n_embd, cfg.block_size)
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)
        self.lm_head.weight = self.emb.tok.weight  # weight tying: the same matrix reads and writes tokens

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.ln_f(self.blocks(self.emb(idx)))
        logits = self.lm_head(x)  # (B, T, vocab)
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --- Milestone 4: Set up the training loop and train the GPT model on the medical text data.
# Hint: Optimizer setup, training loop, loss calculation.

def get_batch(data: torch.Tensor, block_size: int, batch_size: int, gen: torch.Generator | None = None):
    """Random windows: x is tokens i..i+T-1 and y is the same window shifted one to the right (the next token)."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate_loss(model: GPT, data: torch.Tensor, batches: int = 20, batch_size: int = 32, seed: int = 0) -> float:
    """Mean cross-entropy (nats per token) on fixed random windows of `data`."""
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    dev = next(model.parameters()).device
    losses = [model(*(t.to(dev) for t in get_batch(data, model.cfg.block_size, batch_size, gen)))[1].item()
              for _ in range(batches)]
    model.train()
    return sum(losses) / len(losses)


def lr_at(step: int, steps: int, lr: float, warmup: int = 50, min_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay down to min_ratio * lr."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def train(model: GPT, train_ids: torch.Tensor, val_ids: torch.Tensor, steps: int, batch_size: int = 32,
          lr: float = 3e-3, weight_decay: float = 0.1, eval_every: int = 100, device: str = "cpu",
          seed: int = 0, log=print, on_eval=None) -> list[dict]:
    """AdamW + cosine schedule + gradient clipping. Returns the log rows {step, train_loss, val_loss, secs}."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(device).train()
    decay = [p for p in model.parameters() if p.dim() >= 2]       # weight matrices decay, biases and norms do not
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95))
    rows, t0 = [], time.time()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "train_loss": round(evaluate_loss(model, train_ids, 10), 4),
                   "val_loss": round(evaluate_loss(model, val_ids, 10), 4), "secs": round(time.time() - t0, 1)}
            rows.append(row)
            log(f"step {step:5d}  train {row['train_loss']:.3f}  val {row['val_loss']:.3f}  {row['secs']:.0f}s")
            if on_eval:
                on_eval(row)
        if step == steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        x, y = (t.to(device) for t in get_batch(train_ids, model.cfg.block_size, batch_size, gen))
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return rows


def save_model(model: GPT, tok: BPETokenizer, folder) -> None:
    """Write folder/model.pt (config + weights) and folder/tokenizer.json, so nothing needs retraining."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(model.cfg), "state": {k: v.cpu() for k, v in model.state_dict().items()}},
               folder / "model.pt")
    tok.save(folder / "tokenizer.json")


def load_model(folder, device: str = "cpu") -> tuple[GPT, BPETokenizer]:
    folder = Path(folder)
    ckpt = torch.load(folder / "model.pt", map_location=device, weights_only=True)
    model = GPT(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["state"])
    return model.to(device).eval(), BPETokenizer.load(folder / "tokenizer.json")


class NgramBaseline:
    """Interpolated Witten-Bell n-gram over the same BPE tokens. The yardstick the GPT has to beat."""

    def __init__(self, order: int, vocab_size: int):
        self.order, self.V = order, vocab_size
        self.counts = [defaultdict(Counter) for _ in range(order)]  # counts[k][context of k tokens][next token]

    def fit(self, ids: list[int]) -> "NgramBaseline":
        for k in range(self.order):
            for i in range(k, len(ids)):
                self.counts[k][tuple(ids[i - k:i])][ids[i]] += 1
        return self

    def prob(self, context: list[int], tok: int) -> float:
        p = 1.0 / self.V
        for k in range(self.order):                        # lowest order first, then refine with longer contexts
            if k > len(context):
                break
            c = self.counts[k].get(tuple(context[len(context) - k:]) if k else ())
            if not c:
                break
            total, types = sum(c.values()), len(c)
            lam = total / (total + types)
            p = lam * c.get(tok, 0) / total + (1 - lam) * p
        return p

    def distribution(self, context: list[int]) -> list[float]:
        return [self.prob(context, t) for t in range(self.V)]

    def nll(self, ids: list[int]) -> float:
        """Mean negative log-likelihood (nats per token) of ids[1:]."""
        n = self.order
        return -sum(math.log(self.prob(ids[max(0, i - n + 1):i], ids[i])) for i in range(1, len(ids))) / (len(ids) - 1)


def bits_per_char(nats_per_token: float, n_tokens: int, n_chars: int) -> float:
    """Tokenizer-independent score: total bits divided by characters, so different vocabularies compare fairly."""
    return nats_per_token * n_tokens / n_chars / math.log(2)


# --- Milestone 5: Implement a text generation function to complete medical phrases or definitions.
# Hint: Sampling strategies, text generation loop.

def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Top-k keeps the k best tokens; top-p keeps the smallest set whose probabilities sum to p. Others get -inf."""
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        drop = probs.cumsum(-1) - probs >= top_p  # the mass before this token already reaches p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


@torch.no_grad()
def generate_ids(model: GPT, ids: list[int], max_new_tokens: int = 60, temperature: float = 0.8,
                 top_k: int | None = 40, top_p: float | None = 0.9, repetition_penalty: float = 1.2,
                 stop=None, seed: int | None = None) -> tuple[list[int], float]:
    """The generation loop: predict, filter, sample one token, append, repeat.

    temperature <= 0 means greedy. `stop(new_ids) -> bool` ends generation early. Returns (new ids, mean log-prob)."""
    model.eval()
    gen = torch.Generator().manual_seed(seed) if seed is not None else None
    dev = next(model.parameters()).device
    seq, new, logps = list(ids), [], []
    for _ in range(max_new_tokens):
        x = torch.tensor([seq[-model.cfg.block_size:]], device=dev)
        logits = model(x)[0][0, -1].float().cpu()
        if repetition_penalty != 1.0:  # make tokens already used in this completion less likely
            for t in set(new):
                logits[t] = logits[t] / repetition_penalty if logits[t] > 0 else logits[t] * repetition_penalty
        if temperature <= 0:
            nxt = int(torch.argmax(logits))
        else:
            probs = F.softmax(filter_logits(logits / temperature, top_k, top_p), dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        logps.append(float(F.log_softmax(logits, dim=-1)[nxt]))
        seq.append(nxt)
        new.append(nxt)
        if stop and stop(new):
            break
    return new, (sum(logps) / len(logps) if logps else 0.0)


@dataclass
class Completion:
    text: str
    score: float  # mean log-probability per token under the model; higher is a more confident completion


def _trim(text: str, max_words: int | None) -> str:
    """Cut at a paragraph break, at max_words, then back to the last full sentence if there is one."""
    text = text.lstrip("\n").split("\n")[0].rstrip()
    words = text.split(" ")
    if max_words and len(words) > max_words:
        text = " ".join(words[:max_words])
    m = re.match(r"(.*[.;?!])(?=\s|$)", text, re.S)
    if m and len(m.group(1)) >= 0.5 * len(text):
        text = m.group(1)
    return text.rstrip()


def _prompt_ids(tok: BPETokenizer, prompt: str) -> tuple[list[int], bool]:
    """Words carry their leading space in the vocabulary (' muscle'), so a prompt that ends in a space is encoded
    without it and the model's next token supplies it. Otherwise the stray ' ' token would be off-distribution."""
    trailing = prompt.endswith(" ")
    return tok.encode(prompt[:-1] if trailing else prompt, skip_unknown=True) or [0], trailing


def complete(model: GPT, tok: BPETokenizer, prompt: str, max_words: int = 30, n: int = 1, temperature: float = 0.8,
             top_k: int | None = 40, top_p: float | None = 0.9, repetition_penalty: float = 1.2,
             seed: int | None = None) -> list[Completion]:
    """Continue `prompt`. Returns n distinct completions (the text after the prompt), best score first.
    max_words is the length control: generation stops once the completion has that many words."""
    ids, trailing_space = _prompt_ids(tok, prompt)
    max_tokens = int(max_words * 3) + 8

    def done(new):
        text = tok.decode(new)
        return "\n" in text.lstrip("\n") or len(text.split()) > max_words

    out: dict[str, Completion] = {}
    for i in range(n * 3):  # a few spare draws so near-duplicates do not leave us short
        new, score = generate_ids(model, ids, max_tokens, temperature, top_k, top_p, repetition_penalty, done,
                                  None if seed is None else seed + i)
        text = _trim(tok.decode(new), max_words)
        if trailing_space:
            text = text.lstrip(" ")  # the prompt already ends with the space this completion started with
        if text and text not in out:
            out[text] = Completion(text, score)
        if len(out) == n:
            break
    return sorted(out.values(), key=lambda c: -c.score)


@torch.no_grad()
def next_token_candidates(model: GPT, tok: BPETokenizer, prompt: str, k: int = 5) -> list[tuple[str, float]]:
    """The k most likely next tokens and their probabilities, e.g. to finish the word being typed."""
    ids, _ = _prompt_ids(tok, prompt)
    model.eval()
    dev = next(model.parameters()).device
    probs = F.softmax(model(torch.tensor([ids[-model.cfg.block_size:]], device=dev))[0][0, -1].float(), dim=-1)
    top = torch.topk(probs, k)
    return [(tok.itos[int(i)], float(p)) for p, i in zip(top.values, top.indices)]


def define_prompt(term: str) -> str:
    """The prompt that makes the model answer in dictionary style: the term, then its definition."""
    return term.strip().upper() + ". "


# --- Milestone 6: Create a CLI for students to enter partial phrases and get completions.
# Hint: User input handling, command-line interaction.
# Study tools that go beyond completion, built for a medical student.

# Greek and Latin building blocks: prefix (p-), combining form (c), suffix (-s).
ROOTS = {
    "a": "without, not", "an": "without, not", "anti": "against", "brady": "slow", "dys": "bad, difficult, painful",
    "endo": "within", "epi": "upon, above", "hemi": "half", "hyper": "excessive, above", "hypo": "deficient, below",
    "inter": "between", "intra": "within", "macro": "large", "micro": "small", "peri": "around", "poly": "many",
    "post": "after", "pre": "before", "sub": "under", "tachy": "fast", "trans": "across", "dia": "through",
    "arthr": "joint", "cardi": "heart", "cephal": "head", "cerebr": "brain", "chol": "bile", "cyst": "bladder, sac",
    "cyt": "cell", "derm": "skin", "enter": "intestine", "gastr": "stomach", "gluc": "sugar", "hem": "blood",
    "hemat": "blood", "hepat": "liver", "hist": "tissue", "my": "muscle", "myel": "marrow, spinal cord",
    "nephr": "kidney", "neur": "nerve", "oste": "bone", "ot": "ear", "path": "disease", "phleb": "vein",
    "pneum": "lung, air", "pulmon": "lung", "ren": "kidney", "rhin": "nose", "thromb": "clot", "vas": "vessel",
    "ven": "vein", "arteri": "artery", "bronch": "airway", "cost": "rib", "crani": "skull", "gloss": "tongue",
    "laryng": "voice box", "mast": "breast", "ophthalm": "eye", "pharyng": "throat", "thorac": "chest",
    "tonsill": "tonsil", "ur": "urine", "uter": "uterus", "o": "(connecting vowel)",
    "algia": "pain", "ectomy": "surgical removal", "emia": "blood condition", "gram": "record", "graphy": "recording",
    "itis": "inflammation", "logy": "study of", "malacia": "softening", "megaly": "enlargement", "oma": "tumor",
    "osis": "abnormal condition", "pathy": "disease", "penia": "deficiency", "plasty": "surgical repair",
    "rrhage": "bursting forth", "rrhea": "flow", "sclerosis": "hardening", "scope": "instrument to view",
    "stomy": "creation of an opening", "tomy": "cutting into", "uria": "urine condition", "ic": "relating to",
    "al": "relating to", "ous": "relating to", "ia": "condition",
}


def explain_term(term: str) -> list[tuple[str, str]]:
    """Split a medical word into known Greek/Latin parts, e.g. gastroenteritis -> gastr + o + enter + itis.
    Picks the split that explains the most letters. Unknown stretches come back with the meaning '?'."""
    word = re.sub(r"[^a-z]", "", term.lower())
    best: dict[int, tuple[int, list[tuple[str, str]]]] = {}

    def solve(i: int) -> tuple[int, list[tuple[str, str]]]:  # (letters explained, parts) for word[i:]
        if i >= len(word):
            return 0, []
        if i in best:
            return best[i]
        skip_score, skip_parts = solve(i + 1)  # leave this letter unexplained
        res = (skip_score, [(word[i], "?")] + skip_parts)
        for j in range(i + 1, len(word) + 1):
            part = word[i:j]
            if part in ROOTS and (len(part) > 1 or part == "o"):
                score, parts = solve(j)
                score += j - i if part != "o" else 0
                if score > res[0] or (score == res[0] and len(parts) + 1 < len(res[1])):
                    res = (score, [(part, ROOTS[part])] + parts)
        best[i] = res
        return res

    parts: list[tuple[str, str]] = []
    for p, m in solve(0)[1]:  # merge neighbouring unknown letters into one stretch
        if p == "o" and m == "?" and parts:
            parts.append((p, ROOTS["o"]))
        elif m == "?" and parts and parts[-1][1] == "?":
            parts[-1] = (parts[-1][0] + p, "?")
        else:
            parts.append((p, m))
    return parts


_STOP = set("a an and are as at be by for from in is it its of on or that the this to which with when who has have not".split())


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in _STOP}


def overlap_score(answer: str, reference: str) -> float:
    """Share of the reference's content words that the answer also uses (0 to 1). A quick self-check, not a grade."""
    ref = content_words(reference)
    return len(ref & content_words(answer)) / len(ref) if ref else 0.0


def pick_quiz_entries(glossary: list[tuple[str, str]], n: int = 5, seed: int | None = None) -> list[tuple[str, str]]:
    return random.Random(seed).sample(glossary, min(n, len(glossary)))


class History:
    """Completions the student chose to keep, stored with app.storage.Store (one JSON file)."""

    def __init__(self, store):
        self.store = store

    def add(self, prompt: str, completion: str) -> int:
        items = self.store.get("saved", [])
        items.append({"prompt": prompt, "completion": completion})
        self.store.put("saved", items)
        return len(items)

    def items(self) -> list[dict]:
        return self.store.get("saved", [])

    def clear(self) -> None:
        self.store.put("saved", [])


def tutor_explain(term: str, chat=None) -> str:
    """Optional: ask the chat model configured in .env (Ollama, Groq or Gemini; see app/llm.py) for a short explanation.
    It sits beside the from-scratch GPT, which cannot explain terms reliably. The root breakdown is passed along as a hint."""
    if chat is None:
        from app import llm
        chat = llm.chat
    parts = ", ".join(f"{p} = {m}" for p, m in explain_term(term) if m != "?")
    messages = [
        {"role": "system", "content": "You are a concise medical terminology tutor. In at most 4 sentences, explain the term "
                                      "for a first-year student: its meaning, what its word parts mean, and one example of use. "
                                      "If you are not sure, say so."},
        {"role": "user", "content": f"Term: {term}\nWord parts found: {parts or 'none'}"},
    ]
    return chat(messages, temperature=0.2, max_tokens=300).strip()


DEFAULT_MODEL = "models"
