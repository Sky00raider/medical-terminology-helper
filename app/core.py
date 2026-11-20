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
