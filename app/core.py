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
    ids = tok.encode(token, skip_unknown=True)
    if len(ids) != 1:
        raise ValueError(f"{token!r} is not a single token (it splits into {tok.tokens(token)})")
    w = F.normalize(emb_weight.detach(), dim=-1)
    sims = w @ w[ids[0]]
    sims[ids[0]] = -2
    top = torch.topk(sims, k)
    return [(tok.itos[int(i)], float(s)) for s, i in zip(top.values, top.indices)]
