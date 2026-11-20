"""Measure the trained GPT against n-gram baselines on held-out text and held-out dictionary entries.
Writes results/results.md.   Run: python scripts/evaluate.py   (needs models/ from scripts/train.py; about 3 minutes on a CPU)"""
import csv
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import core  # noqa: E402

N_PROBE = 80          # held-out dictionary entries used for the definition test
SAMPLES = ["Inflammation of the", "The femur is", "GASTRITIS. ", "HYPERTENSION. ", "The function of the kidney",
           "The three layers of the heart wall", "The cerebrum", "A fracture of the", "Blood is carried from the heart by",
           "PERICARDITIS. ", "The pancreas secretes", "ANEURYSM. "]
TERMS = ["gastrocnemius", "pericarditis", "thrombocytopenia", "nephrectomy", "hemoglobin", "cholecystitis"]


def real_word_rate(text: str, vocab: set[str]) -> float:
    words = [w.lower() for w in core.re.findall(r"[A-Za-z]+", text)]
    return sum(w in vocab for w in words) / max(1, len(words))


def ngram_complete(ng, tok, prompt: str, max_tokens: int = 60) -> str:
    """Greedy continuation under the n-gram baseline, stopping at the same sentence end the GPT output is trimmed at."""
    ids, _ = core._prompt_ids(tok, prompt)
    new = []
    for _ in range(max_tokens):
        dist = ng.distribution((ids + new)[-(ng.order - 1):])
        new.append(max(range(len(dist)), key=dist.__getitem__))
        if len(tok.decode(new).split()) > 30:
            break
    text = core._trim(tok.decode(new), 25)
    return text.lstrip(" ") if prompt.endswith(" ") else text


def main() -> None:
    t0 = time.time()
    model, tok = core.load_model(ROOT / "models")
    torch.manual_seed(0)
    train_text = (ROOT / "data" / "train.txt").read_text(encoding="utf-8")
    val_text = (ROOT / "data" / "val.txt").read_text(encoding="utf-8")
    train_ids, val_ids = tok.encode(train_text), tok.encode(val_text, skip_unknown=True)
    vocab = {w.lower() for w in core.re.findall(r"[A-Za-z]+", train_text)}
    B = model.cfg.block_size
    chunks = [val_ids[i:i + B + 1] for i in range(0, len(val_ids) - B - 1, B + 1)]   # same windows for every model
    n_pred, n_chars = len(chunks) * B, len(val_text) * len(chunks) * (B + 1) / len(val_ids)
    lines = ["# Results", "", f"Model: {model.n_params():,} parameters, vocab {tok.vocab_size}, context {B} tokens. "
             f"Evaluation text: {len(chunks)} held-out windows of {B + 1} BPE tokens ({n_pred:,} predictions) from `data/val.txt`, "
             "text the model never saw in training. Every model scores the same windows with the same tokenizer.", ""]

    # 1. Language-model quality: tokenizer-independent bits per character, plus next-token accuracy
    x = torch.tensor(chunks)
    with torch.no_grad():
        logits = torch.cat([model(x[i:i + 64, :-1])[0] for i in range(0, len(x), 64)])
    targets = x[:, 1:]
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1)).item()
    top5 = logits.topk(5, dim=-1).indices
    gpt_top1 = (top5[..., 0] == targets).float().mean().item()
    gpt_top5 = (top5 == targets.unsqueeze(-1)).any(-1).float().mean().item()
    rows = [("GPT (this project)", nll, gpt_top1, gpt_top5)]
    probe = random.Random(0).sample(range(len(chunks)), min(len(chunks), 25))
    n_acc = len(probe) * len(range(1, B, 4))
    for order in (1, 2, 3, 4):
        ng = core.NgramBaseline(order, tok.vocab_size).fit(train_ids)
        nll_ng = sum(ng.nll(c) * (len(c) - 1) for c in chunks) / n_pred
        hits1 = hits5 = total = 0
        for ci in probe:   # accuracy on a subsample: the n-gram needs a full distribution per position
            for i in range(1, B, 4):
                d = ng.distribution(chunks[ci][max(0, i - order + 1):i])
                top = sorted(range(len(d)), key=d.__getitem__, reverse=True)[:5]
                hits1 += top[0] == chunks[ci][i]
                hits5 += chunks[ci][i] in top
                total += 1
        rows.append((f"{order}-gram" + (" (unigram)" if order == 1 else ""), nll_ng, hits1 / total, hits5 / total))
    lines += ["## 1. Predicting held-out text", "",
              "| model | loss (nats/token) | bits per character | next-token top-1 | top-5 |", "|---|---|---|---|---|"]
    for name, l, a1, a5 in rows:
        lines.append(f"| {name} | {l:.3f} | {core.bits_per_char(l, n_pred, n_chars):.3f} | {a1:.1%} | {a5:.1%} |")
    lines += ["", f"Uniform guessing would score {math.log(tok.vocab_size):.2f} nats per token. Bits per character divides the total "
              "bits by the characters covered, so scores stay comparable if the tokenizer changes. The n-gram accuracies use a "
              f"subsample of {n_acc} positions (the same positions for every order).", ""]
    ng4 = core.NgramBaseline(4, tok.vocab_size).fit(train_ids)

    # 2. Defining held-out terms
    glossary = core.load_glossary(ROOT / "data" / "glossary.json")
    entries = random.Random(1).sample(glossary, min(N_PROBE, len(glossary)))
    res = {"GPT greedy": [], "GPT sampled (T=0.7)": [], "4-gram greedy": []}
    shown = []
    for i, (term, ref) in enumerate(entries):
        p = core.define_prompt(term)
        g = core.complete(model, tok, p, max_words=25, temperature=0, repetition_penalty=1.0)[0].text
        s = core.complete(model, tok, p, max_words=25, temperature=0.7, seed=i)
        s = s[0].text if s else ""
        n = ngram_complete(ng4, tok, p)
        for key, text in zip(res, (g, s, n)):
            res[key].append((core.overlap_score(text, ref), real_word_rate(text, vocab), len(text.split())))
        if i < 6:
            shown.append((term, ref, g, n))
    lines += ["## 2. Defining held-out dictionary terms", "",
              f"{len(entries)} entries from Gould's dictionary that were held out of training. Prompt: `TERM. `; the completion "
              "is scored against the real definition. *Key-word overlap* is the share of the reference's content words the "
              "completion also uses. *Real-word rate* is the share of generated words that occur in the training corpus.", "",
              "| method | key-word overlap | real-word rate | mean words |", "|---|---|---|---|"]
    for key, vals in res.items():
        lines.append(f"| {key} | {sum(v[0] for v in vals) / len(vals):.1%} | {sum(v[1] for v in vals) / len(vals):.1%} | "
                     f"{sum(v[2] for v in vals) / len(vals):.1f} |")
    lines += ["", "Side by side (reference, GPT greedy, 4-gram greedy):", ""]
    for term, ref, g, n in shown:
        lines += [f"- **{term}**", f"  - reference: {ref}", f"  - GPT: {g}", f"  - 4-gram: {n}"]

    # 3. Sample completions
    lines += ["", "## 3. Sample completions (temperature 0.7, seed 0)", ""]
    for p in SAMPLES:
        c = core.complete(model, tok, p, max_words=25, n=1, temperature=0.7, seed=0)
        lines.append(f"- `{p}` -> {c[0].text if c else '(nothing)'}")

    # 4. Tokenizer
    lines += ["", "## 4. What the BPE tokenizer learned", "",
              f"Vocabulary {tok.vocab_size}; the training text is {len(train_ids) / len(train_text):.3f} tokens per character "
              f"({len(train_text) / len(train_ids):.2f} characters per token).", "", "| term | tokens |", "|---|---|"]
    lines += [f"| {t} | {' · '.join(tok.tokens(' ' + t))} |" for t in TERMS]

    # 5. Embedding neighbours
    lines += ["", "## 5. Nearest neighbours in the learned embedding space (cosine)", ""]
    for t in ("itis", " heart", " kidney", " artery", " fever"):
        try:
            near = core.nearest_tokens(tok, model.emb.tok.weight, t, 6)
            lines.append(f"- `{t}`: " + ", ".join(f"`{n}` {s:.2f}" for n, s in near))
        except ValueError:
            pass

    # 6. Training curve
    loss_csv = ROOT / "results" / "loss.csv"
    if loss_csv.exists():
        rows_l = list(csv.DictReader(open(loss_csv)))
        lines += ["", "## 6. Training curve", "", "| step | train loss | val loss | seconds |", "|---|---|---|---|"]
        pick = rows_l[:: max(1, len(rows_l) // 8)]
        if pick[-1] is not rows_l[-1]:
            pick.append(rows_l[-1])
        lines += [f"| {r['step']} | {float(r['train_loss']):.3f} | {float(r['val_loss']):.3f} | {r['secs']} |" for r in pick]

    lines += ["", f"_Generated by `scripts/evaluate.py` in {time.time() - t0:.0f}s._"]
    (ROOT / "results" / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
